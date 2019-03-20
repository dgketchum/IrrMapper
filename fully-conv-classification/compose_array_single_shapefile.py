# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2. (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================================

import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
import pickle
from copy import deepcopy
from warnings import warn
from datetime import datetime
from fiona import open as fopen
from numpy import linspace, max, nan, unique, ndarray, swapaxes, zeros, asarray
import h5py
from numpy.random import shuffle
from pandas import DataFrame, Series
import warnings
from pyproj import Proj, transform
from rasterio import open as rasopen
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
from shapefile_utils import get_shapefile_path_row
loc = os.path.dirname(__file__)
WRS_2 = loc.replace('pixel_classification',
        os.path.join('spatial_data', 'wrs2_usa_descending.shp'))

'''
This script contains a class meant to gather data from rasters using a polygon shapefile.
The high-level method `extract_sample` will return an object ready for a
learning algorithm.
'''
loc = os.path.dirname(__file__)
WRS_2 = loc.replace('pixel_classification',
        os.path.join('spatial_data', 'wrs2_descending.shp'))

class NoCoordinateReferenceError(Exception):
    pass


class UnexpectedCoordinateReferenceSystemError(Exception):
    pass


class ShapefileSamplePoints:

    def __init__(self, shapefile_path=None, sample_point_directory=None, m_instances=None):
        self.outfile = os.path.splitext(shapefile_path)[0]
        self.outfile += "_sample_points.shp"
        if sample_point_directory:
            self.outfile = os.path.join(sample_point_directory, self.outfile)

        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE'])
        self.m_instances = m_instances
        self.object_id = 0
        self.shapefile_path = shapefile_path
        self.path, self.row = get_shapefile_path_row(shapefile_path)

    def _random_points(self, coords):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * self.m_instances)
        y_range = linspace(min_y, max_y, num=2 * self.m_instances)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):
        # TODO: Encode class_code in shapefile schema.
        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
            'X': coord[0],
            'Y': coord[1],
            'POINT_TYPE': val},
            ignore_index=True)
        self.object_id += 1

    def save_sample_points(self):

        points_schema = {
                'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10')]),
                'geometry': 'Point'}
        meta = self.tile_geometry.copy()
        meta['schema'] = points_schema
            
        with fopen(self.outfile, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']), ('POINT_TYPE', row['POINT_TYPE'])])
                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                    'geometry': mapping(pt)})
        return None

    def _get_polygons(self, vector):
        with fopen(vector, 'r') as src:
            crs = src.crs
            if not crs:
                raise NoCoordinateReferenceError(
                        'Provided shapefile has no reference data.')
                if crs['init'] != 'epsg:4326':
                    raise UnexpectedCoordinateReferenceSystemError(
                            'Provided shapefile should be in unprojected (geographic)'
                            'coordinate system, i.e., WGS84, EPSG 4326, {} is not'.format(
                                vector))
            clipped = src.filter(mask=self.tile_bbox)
            polys = []
            bad_geo_count = 0
            for feat in clipped:
                try:
                    geo = shape(feat['geometry'])
                    polys.append(geo)
                except AttributeError:
                    bad_geo_count += 1

        return polys

    def create_sample_points(self, save_points=True):
        """ Create a clipped training set from polygon shapefiles.

        This complicated-looking function finds the wrs_2 descending Landsat tile corresponding
        to the path row provided, gets the bounding box and profile (aka meta) from
        compose_array.get_tile_geometry, clips the training data to the landsat tile, then perform
        s a union to reduce the number of polygon objects.
        The dict object this uses has a template in pixel_classification.runspec.py.
        Approach is to loop through the polygons, create a random grid of points over the
        extent of each polygon, random shuffle order of points, loop over points, check if
        point is within polygon, and if within, create a sample point.

        If a relatively simple geometry is available, use create_negative_sample_points(), though if
        there are > 10**4 polygons, it will probably hang on unary_union(). """

        polygons = self._get_polygons(self.shapefile_path)
        instance_count = 0
        print("Making sample points. You have {} polygons".format(len(polygons)))
        print("N_instances:", self.m_instances)

        if len(polygons) > self.m_instances:
            areas = zip(polygons, [x.area for x in polygons])
            srt = sorted(areas, key=lambda x: x[1], reverse=True)
            polygons = [x for x, y in srt[:self.m_instances]]

        if not isinstance(polygons, list):
            polygons = [polygons] # for the case of a single polygon.

        positive_area = sum([x.area for x in polygons]) # the sum of all
        # the areas.
        class_count = 0

        for i, poly in enumerate(polygons):
            if class_count >= self.m_instances:
                break
            fractional_area = poly.area / positive_area # percent of
            # total area that this polygon occupies
            required_points = max([1, fractional_area * self.m_instances]) # how
            # many points overall that are required to evenly
            # sample from each polygon, based on area.
            poly_pt_ct = 0
            x_range, y_range = self._random_points(poly.bounds)
            for coord in zip(x_range, y_range):
                if instance_count >= self.m_instances:
                    break
                if Point(coord[0], coord[1]).within(poly): 
                    self._add_entry(coord)
                    poly_pt_ct += 1
                    instance_count += 1
                    # print(instance_count)
                if poly_pt_ct >= required_points:
                    break
            class_count += poly_pt_ct

        if save_points:
            self.save_sample_points()

    @property
    def tile_bbox(self):
        with fopen(WRS_2, 'r') as wrs:
            for feature in wrs:
                fp = feature['properties']
                if fp['PATH'] == self.path and fp['ROW'] == self.row:
                    bbox = feature['geometry']
                    return bbox

    def _get_crs(self):
        for key, val in self.paths_map.items():
            with rasopen(val, 'r') as src:
                crs = src.crs
            break
        return crs

    @property
    def tile_geometry(self):
        with fopen(WRS_2, 'r') as wrs:
            wrs_meta = wrs.meta.copy()
        return wrs_meta

class PTASingleShapefile:
    # TODO: This class can be easily modified 
    # for the extraction of pixel-wise data from landsat images
    # on a large scale.
    def __init__(self, master_raster=None, shapefile_path=None, class_code=None, path=None,
            row=None, masked_raster=None, training_directory=None, paths_map=None, masks=None,
            instances=None, sz=1000, overwrite_points=None, kernel_size=None, data_filename=None):
        self.shapefile_path = shapefile_path
        self.path = path
        self.object_id = 0
        self.data_filename = data_filename 
        self.paths_map = paths_map
        self.masks = masks
        self.row = row
        self.training_directory = training_directory
        self.overwrite_points=overwrite_points
        self.class_code = class_code
        self.crs = self._get_crs()
        self.m_instances = instances
        self.sz = sz
        self.master_raster = master_raster
        self.data = None
        self.kernel_size = kernel_size
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE'])

    def extract_sample(self, save_points=True):
        # TODO: Pare down this class' methods. 
        # Because of the large data size, pickling output data
        # (and therefore using a one-band at a time extraction approach)
        # is not feasible. 

        out = os.path.splitext(self.shapefile_path)[0]
        out += "_sample_points.shp"
        if os.path.isfile(out):
            print("sample points already created")
            self._populate_array_from_points(out)
        else:
            print("Sample points not detected at {}".format(out))
        if self.master_raster is not None:
            self.training_data_from_master_raster()
        else:
            self.populate_raster_data_array()

    def _populate_array_from_points(self, fname):

        with fopen(fname, 'r') as src:
            for feat in src:
                coords = feat['geometry']['coordinates']
                val = feat['properties']['POINT_TYPE']
                self._add_entry(coords, val=val)

    def _dump_data(self, data):
        n = "class_{}_train.h5".format(self.class_code)
        if self.data_filename is None:
            to_save = os.path.join(self.training_directory, n)
        else:
            to_save = self.data_filename
        with h5py.File(to_save, 'a') as f:
            pref = os.path.basename(self.shapefile_path)
            dset = f.create_dataset("{}_{}".format(pref,
                str(datetime.now())), data=data)

    def training_data_from_master_raster(self):

        ofs = self.kernel_size // 2
        sz = self.sz # some heuristic that indicates when I run out of memory
        tmp_arr = []
        with rasopen(self.master_raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            affine = rsrc.transform

        for ind, row in self.extracted_points.iterrows():
            # iterate through extracted points.
            if (ind+1) % sz == 0:
                print("Writing to disk...")
                qq = asarray(tmp_arr)
                del tmp_arr
                self._dump_data(qq)
                del qq
                tmp_arr = []

            x, y = self._geo_point_to_projected_coords(row['X'], row['Y'])
            c, r = ~affine * (x, y)
            try:
                rr = int(r); cc = int(c)
                raster_subgrid = rass_arr[:, rr-ofs:rr+ofs+1, cc-ofs:cc+ofs+1]
                tmp_arr.append(raster_subgrid)

            except IndexError as e:
                print(e)

        if len(tmp_arr): 
            print("Writing to disk...") 
            qq = asarray(tmp_arr)
            self._dump_data(qq)
            del qq
            del tmp_arr

    def populate_raster_data_array(self, save=True):

        for key, val in self.paths_map.items():
            s = self._grid_raster_extract(val, _name=key)
            print('Extracting {}'.format(key))
            self.extracted_points = self.extracted_points.join(s, how='outer')

        for key, val in self.masks.items():
            s = self._grid_raster_extract(val, _name=key)
            print('Extracting {}'.format(key))
            self.extracted_points = self.extracted_points.join(s, how='outer')

        data_array, targets = self._purge_raster_array()
        data = {'df': data_array,
                'features': data_array.columns.values,
                'data': data_array.values,
                'target_values': targets,
                'paths_map': self.paths_map}

        print('feature dimensions: {}'.format(data_array.shape))

        for key, val in data.items():
            setattr(self, key, val)

    def _purge_raster_array(self):
        data_array = deepcopy(self.extracted_points)
        target_vals = Series(data_array.POINT_TYPE.values, name='POINT_TYPE')
        data_array.drop(['X', 'Y', 'FID', 'POINT_TYPE'], axis=1, inplace=True)
        try:
            for msk in self.masks.keys():
                for idx, sub_raster in enumerate(data_array[msk]):
                    if sub_raster[self.kernel_size // 2][self.kernel_size // 2] == 1.:
                        data_array.loc[idx, :] = nan # make whole row NaN
        except TypeError as e:
            print(sub_raster, msk, idx)
            data_array.loc[idx, :] = nan

        try:
            for bnd in self.paths_map.keys():
                for idx, sub_raster in enumerate(data_array[bnd]):
                    if sub_raster[self.kernel_size // 2][self.kernel_size // 2] == 0.:
                        data_array.loc[idx, :] = nan
        except TypeError as e:
            data_array.loc[idx, :] = nan

        data_array = data_array.join(target_vals, how='outer')

        data_array.dropna(axis=0, inplace=True)
        data_array.drop(self.masks, axis=1, inplace=True)
        target_vals = data_array.POINT_TYPE.values

        data_array = data_array.drop(['POINT_TYPE'],
                axis=1, inplace=False)
        return data_array, target_vals

    def _geo_point_to_projected_coords(self, x, y):

        in_crs = Proj(init='epsg:4326')
        out_crs = Proj(init=self.crs['init'])
        x, y = transform(in_crs, out_crs, x, y)
        return x, y

    def _grid_raster_extract(self, raster, _name):
        """
        Open the raster. Store the points in a Series - a labeled
        numpy array. Then in _purge array, we iterate over the masks
        and the paths_map and drop pixels where masks = 1 and pixels where bound = 0.
        """

        with rasopen(raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            affine = rsrc.transform

        s = Series(index=range(0, self.extracted_points.shape[0]), name=_name, dtype=object)
        for ind, row in self.extracted_points.iterrows():
            x, y = self._geo_point_to_projected_coords(row['X'], row['Y'])
            c, r = ~affine * (x, y)
            try:
                ofs = self.kernel_size // 2
                rr = int(r); cc = int(c)
                raster_subgrid = rass_arr[rr-ofs:rr+ofs+1, cc-ofs:cc+ofs+1] # possible issues: edges of image
                s[ind] = raster_subgrid
            except IndexError:
                s[ind] = None

        return s

    @property
    def tile_bbox(self):
        with fopen(WRS_2, 'r') as wrs:
            for feature in wrs:
                fp = feature['properties']
                if fp['PATH'] == self.path and fp['ROW'] == self.row:
                    bbox = feature['geometry']
                    return bbox

    def _get_crs(self):
        for key, val in self.paths_map.items():
            with rasopen(val, 'r') as src:
                crs = src.crs
            break
        return crs

    @property
    def tile_geometry(self):
        with fopen(WRS_2, 'r') as wrs:
            wrs_meta = wrs.meta.copy()
        return wrs_meta

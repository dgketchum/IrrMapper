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

from fiona import open as fopen
from numpy import linspace, max, nan, unique, ndarray, swapaxes, zeros
import h5py
from numpy.random import shuffle
from pandas import DataFrame, Series
import warnings
from pyproj import Proj, transform
from rasterio import open as rasopen
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
loc = os.path.dirname(__file__)
WRS_2 = loc.replace('pixel_classification',
                    os.path.join('spatial_data', 'wrs2_descending.shp'))

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


class PTASingleShapefile:

    def __init__(self, master_raster=None, shapefile_path=None, class_code=None, path=None, 
            row=None, masked_raster=None, training_directory=None, paths_map=None, masks=None, 
            instances=None, overwrite_points=None, kernel_size=None):
        self.shapefile_path = shapefile_path
        self.path = path
        self.object_id = 0
        self.paths_map = paths_map
        self.masks = masks
        self.row = row
        self.training_directory = training_directory
        self.overwrite_points=overwrite_points
        self.class_code = class_code
        self.crs = self._get_crs()
        self.m_instances = instances
        self.master_raster = master_raster
        self.masked_raster = masked_raster
        if masked_raster is not None:
            print(masked_raster, "Masked raster present.")
        self.data = None
        self.kernel_size = kernel_size
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE'])

    def extract_sample(self, save_points=True):

        out = os.path.splitext(self.shapefile_path)[0]
        out += "_sample_points.shp"
        if os.path.isfile(out) and not self.overwrite_points:
            print("sample points already created")
            self._populate_array_from_points(out)
        else:
            self.create_sample_points()
            if save_points:
                self.save_sample_points()
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

    def _create_hdf_file(self):
        n = os.path.basename(self.shapefile_path)[:-4] + "_train.h5"
        to_save = os.path.join(self.training_directory, n)
        hdf5 = h5py.File(to_save, 'a')
        shape = (self.m_instances, self.kernel_size, self.kernel_size, len(self.paths_map.keys()))
        hdf5.create_dataset("cc:{}".format(self.class_code), shape, compression='gzip')
        return hdf5

    def _verify_point(self, x, y):
        """ Check to see if x, y is masked. """
        pass


    def training_data_from_master_raster(self):

        ofs = self.kernel_size // 2
        #TODO: Make saving data easier. 
        # Query how much memory I have left?
 

        sz = 10000 # some heuristic that indicates when I run out of memory
        tmp_arr = zeros((sz, len(self.paths_map.keys()), self.kernel_size, self.kernel_size))

        n = os.path.basename(self.shapefile_path)[:-4] + "_train.h5"
        to_save = os.path.join(self.training_directory, n)
        hdf5 = h5py.File(to_save, 'a')
        shape = (self.m_instances, len(self.paths_map.keys()), self.kernel_size, self.kernel_size)
        dset = hdf5.create_dataset("cc:{}".format(self.class_code), shape, compression='gzip')

        with rasopen(self.master_raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            affine = rsrc.transform

        j = 0
        p = 0
        for ind, row in self.extracted_points.iterrows():
            p = ind
            # iterate through extracted points. 
            if j == sz:
                print("Writing to disk...")
                dset[ind-j:ind, :, :, :] = tmp_arr
                j = 0
            if ind % 1000 == 0:
                print("Step:", ind)
            x, y = self._geo_point_to_projected_coords(row['X'], row['Y'])
            c, r = ~affine * (x, y)
            try:
                rr = int(r); cc = int(c)
                raster_subgrid = rass_arr[:, rr-ofs:rr+ofs+1, cc-ofs:cc+ofs+1] 
                # possible issues: edges of image
                # rs = swapaxes(raster_subgrid, 0, 2) # for N, H, W, C format
                tmp_arr[j, :, :, :] = raster_subgrid
                j += 1

            except IndexError as e:
                print(e)
                tmp_arr[j, :, :, :] = nan
                j += 1

        print("Writing to disk...")
        dset[p-j:p, :, :, :] = tmp_arr
        hdf5.close()

    def create_sample_points(self):
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
        print("Making sample pts")

        polygons = self._get_polygons(self.shapefile_path) 
        instance_count = 0

        if len(polygons) < 2:
            warnings.warn("You have < 2 polygons in shapefile {}.  ".format(os.path.basename(self.shapefile_path), Warning))

        if len(polygons) > self.m_instances:
            areas = zip(polygons, [x.area for x in polygons])
            srt = sorted(areas, key=lambda x: x[1], reverse=True)
            polygons = [x for x, y in srt[:self.m_instances]]

        polygons = unary_union(polygons)
        if not isinstance(polygons, list):
            polygons = [polygons] # for the case of a single polygon.

        positive_area = sum([x.area for x in polygons]) # the sum of all
        # the areas.
        class_count = 0

        for i, poly in enumerate(polygons):
            if class_count >= self.m_instances:
                print("Broke b/c class_count >= m_instances")
                break
            fractional_area = poly.area / positive_area # percent of
            # total area that this polygon occupies 
            required_points = max([1, fractional_area * self.m_instances]) # how
            # many points overall that are required to evenly 
            # sample from each polygon, based on area. 
            poly_pt_ct = 0
            #while poly_pt_ct < required_points: # I wasn't getting enough points.
            x_range, y_range = self._random_points(poly.bounds)
            for coord in zip(x_range, y_range):
                if Point(coord[0], coord[1]).within(poly):
                    self._add_entry(coord, val=self.class_code)
                    poly_pt_ct += 1
                    instance_count += 1
                    if instance_count % 1000 == 0:
                        print("Instances:", instance_count)

                if instance_count > self.m_instances:
                    print("Broke b/c instance_count > m_instances")
                    break

                if poly_pt_ct >= required_points:
                    print("Broke b/c poly_pt_ct > required_points")
                    break

            class_count += poly_pt_ct

        print("Final number of points: ", self.object_id)

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
            print(sub_raster, msk, idx)
            data_array.loc[idx, :] = nan

        data_array = data_array.join(target_vals, how='outer')

        data_array.dropna(axis=0, inplace=True)
        data_array.drop(self.masks, axis=1, inplace=True)
        target_vals = data_array.POINT_TYPE.values

        data_array = data_array.drop(['POINT_TYPE'],
                                     axis=1, inplace=False)
        return data_array, target_vals

    def _random_points(self, coords):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=10 * self.m_instances)
        y_range = linspace(min_y, max_y, num=10 * self.m_instances)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val},
                                                             ignore_index=True)
        self.object_id += 1

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
    
    
    def save_sample_points(self):
 
        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10')]),
            'geometry': 'Point'}
        meta = self.tile_geometry.copy()
        meta['schema'] = points_schema
        
        out = os.path.splitext(self.shapefile_path)[0]
        out += "_sample_points.shp"
     
        with fopen(out, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']), ('POINT_TYPE', row['POINT_TYPE'])])
                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

    @property 
    def tile_geometry(self): 
        with fopen(WRS_2, 'r') as wrs: 
            wrs_meta = wrs.meta.copy() 
        return wrs_meta 

    def to_pickle(self, data, path):

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)

        return path


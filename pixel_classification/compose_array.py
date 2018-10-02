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
from numpy import linspace, max, nan, unique
from numpy.random import shuffle
from pandas import DataFrame, Series
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


class NoCoordinateReferenceError(Exception):
    pass


class UnexpectedCoordinateReferenceSystemError(Exception):
    pass


class PixelTrainingArray(object):
    """
    Notes: The training shape must be un-projected, in the WGS84 EPSG 4326 coordinate reference system.
    For now, images are Landsat images. The 'images' parameter is for a directory of images from
    one path,row Landsat tile.
    """

    def __init__(self, root=None, geography=None, paths_map=None, masks=None,
            instances=None, from_dict=None, pkl_path=None,
            overwrite_array=False, overwrite_points=False):

        """

        :param pkl_path:
        :param overwrite_points:
        :param max_cloud:
        :param training_vectors: in the WGS84 EPSG 4326 coordinate reference system. (str)(.shp)
        :param root: Directory of images from one path,row Landsat tile, use warp_vrt to set them
        at the same geometry. (str)
        :param instances: The number of sample points to extract (int). A small sample size will result
        in an excessive number of 'positive' sample points, as each valid positive sample geometry will
        be sampled once.  As the sample size becomes large (perhaps 10**5), the dataset will approach
        feature balance. Each point is taken from a random spatial index within each polygon.  Approximate
        feature balance is hard-coded in this class.
        :param pkl_path: If the data exists, specify this path to instantiate a data-filled instance
        without repeating the time-consuming sampling process. (bool)
        :param overwrite_array:
        """

        self.overwrite_array = overwrite_array
        self.overwrite_points = overwrite_points

        if pkl_path and not overwrite_array:
            self.from_pickle(pkl_path)
            self.array_exists = True

        elif from_dict:
            self._from_dict(from_dict)
            self.array_exists = True

        else:
            self.array_exists = False
            self.geography = geography
            self.paths_map = paths_map
            self.masks = masks
            self.crs = self._get_crs()
            self.root = root
            self.path_row_dir = os.path.join(self.root, str(geography.path), str(geography.row))
            self.year_dir = os.path.join(self.path_row_dir, str(geography.year))
            self.is_binary = None

            self.features = None
            self.data = None
            self.target_values = None

            self.m_instances = instances
            self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE'])
            self.object_id = 0

    def extract_sample(self, save_points=True):

        if self.array_exists and not self.overwrite_array:
            return None

        if not os.path.isfile(self.shapefile_path):
            self.create_sample_points()

        elif self.overwrite_points:
            self.create_sample_points()

        else:
            self._populate_array_from_points()

        if save_points:
            self.save_sample_points()

        if self.overwrite_array and not self.array_exists:
            self.populate_data_array()

    def create_sample_points(self):
        """ Create a clipped training set from polygon shapefiles.

        This complicated-looking function finds the wrs_2 descending Landsat tile corresponding
        to the path row provided, gets the bounding box and profile (aka meta) from
        compose_array.get_tile_geometry, clips the training data to the landsat tile, then performs a
        union to reduce the number of polygon objects.

        The dict object this uses has a template in pixel_classification.runspec.py.

        Approach is to loop through the polygons, create a random grid of points over the extent of
        each polygon, random shuffle order of points, loop over points, check if point is within polygon,
        and if within, create a sample point.

        If a relatively simple geometry is available, use create_negative_sample_points(), though if
        there ar > 10**4 polygons, it will probably hang on unary_union().

        """

        for class_code, _dict in self.geography.attributes.items():
            print(_dict['ltype'])
            polygons = self._get_polygons(_dict['path'])
            _dict['instance_count'] = 0

            if len(polygons) > self.m_instances:
                areas = zip(polygons, [x.area for x in polygons])
                srt = sorted(areas, key=lambda x: x[1], reverse=True)
                polygons = [x for x, y in srt[:self.m_instances]]

            polygons = unary_union(polygons)
            positive_area = sum([x.area for x in polygons])
            class_count = 0

            for i, poly in enumerate(polygons):
                if class_count >= self.m_instances:
                    break
                fractional_area = poly.area / positive_area
                required_points = max([1, fractional_area * self.m_instances])
                x_range, y_range = self._random_points(poly.bounds)
                poly_pt_ct = 0

                for coord in zip(x_range, y_range):
                    if Point(coord[0], coord[1]).within(poly):
                        self._add_entry(coord, val=class_code)
                        poly_pt_ct += 1
                        _dict['instance_count'] += 1

                    if _dict['instance_count'] > self.m_instances:
                        break

                    if poly_pt_ct >= required_points:
                        break

                class_count += poly_pt_ct

    def populate_data_array(self):

        for key, val in self.paths_map.items():
            s = self._point_raster_extract(val, _name=key)
            print('Extracting {}'.format(key))
            self.extracted_points = self.extracted_points.join(s, how='outer')

        for key, val in self.masks.items():
            s = self._point_raster_extract(val, _name=key)
            print('Extracting {}'.format(key))
            self.extracted_points = self.extracted_points.join(s, how='outer')

        data_array, targets = self._purge_array()
        data = {'df': data_array,
                'features': data_array.columns.values,
                'data': data_array.values,
                'target_values': targets,
                'paths_map': self.paths_map}

        print('feature dimensions: {}'.format(data_array.shape))
        for key, val in data.items():
            setattr(self, key, val)

        self.to_pickle(data)

        self._check_targets(targets)

    def save_sample_points(self):

        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10')]),
            'geometry': 'Point'}
        meta = self.tile_geometry.copy()
        meta['schema'] = points_schema

        with fopen(self.shapefile_path, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']), ('POINT_TYPE', row['POINT_TYPE'])])
                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

    def to_pickle(self, data, path=None):
        if not path:
            path = self.data_path

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)

        return path

    def from_pickle(self, path=None):
        if not path:
            path = self.data_path

        pkl = pickle.load(open(path, 'rb'))
        for key, val in pkl.items():
            setattr(self, key, val)

        self._check_targets(self.target_values)

    def _purge_array(self):

        data_array = deepcopy(self.extracted_points)
        target_vals = Series(data_array.POINT_TYPE.values, name='POINT_TYPE')
        data_array.drop(['X', 'Y', 'FID', 'POINT_TYPE'], axis=1, inplace=True)

        for msk in self.masks.keys():
            data_array[data_array[msk] == 1.] = nan

        for bnd in self.paths_map.keys():
            data_array[data_array[bnd] == 0.] = nan

        data_array = data_array.join(target_vals, how='outer')

        data_array.dropna(axis=0, inplace=True)
        data_array.drop(self.masks, axis=1, inplace=True)
        target_vals = data_array.POINT_TYPE.values

        data_array = data_array.drop(['POINT_TYPE'],
                                     axis=1, inplace=False)
        return data_array, target_vals

    def _from_dict(self, data):

        for key, val in data.items():
            setattr(self, key, val)

        self._check_targets(self.target_values)

    def _check_targets(self, target_vals):

        unique_targets = len(unique(target_vals))

        if not target_vals.any():
            pass
        elif unique_targets == 2:
            self.is_binary = True
        elif 0 < unique_targets < 2:
            warn('This dataset has fewer than two target classes,'
                 'classification is meaningless.')
        elif unique_targets > 2:
            self.is_binary = False
        else:
            warn('This dataset has {} target classes'.format(unique_targets))
            self.is_binary = False

    def _point_raster_extract(self, raster, _name):

        with rasopen(raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            affine = rsrc.transform

        s = Series(index=range(0, self.extracted_points.shape[0]), name=_name)
        for ind, row in self.extracted_points.iterrows():
            x, y = self._geo_point_to_projected_coords(row['X'], row['Y'])
            c, r = ~affine * (x, y)
            try:
                raster_val = rass_arr[int(r), int(c)]
                s[ind] = float(raster_val)
            except IndexError:
                s[ind] = None

        return s

    def _random_points(self, coords):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * self.m_instances)
        y_range = linspace(min_y, max_y, num=2 * self.m_instances)
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

    def _get_crs(self):
        for key, val in self.paths_map.items():
            with rasopen(val, 'r') as src:
                crs = src.crs
            break
        return crs

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

    def _populate_array_from_points(self):
        # TODO: replace with geopandas.shp_to_dataframe
        with fopen(self.shapefile_path, 'r') as src:
            for feat in src:
                coords = feat['geometry']['coordinates']
                val = feat['properties']['POINT_TYPE']
                self._add_entry(coords, val=val)

    @property
    def data_path(self):
        return os.path.join(self.root, 'data.pkl')

    @property
    def shapefile_path(self):
        return os.path.join(self.root, 'sample_points.shp')

    @property
    def model_path(self):
        return os.path.join(self.root, 'model.pkl')

    @property
    def tile_geometry(self):
        with fopen(WRS_2, 'r') as wrs:
            wrs_meta = wrs.meta.copy()
        return wrs_meta

    @property
    def tile_bbox(self):
        with fopen(WRS_2, 'r') as wrs:
            for feature in wrs:
                fp = feature['properties']
                if fp['PATH'] == self.geography.path and fp['ROW'] == self.geography.row:
                    bbox = feature['geometry']
                    return bbox


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================

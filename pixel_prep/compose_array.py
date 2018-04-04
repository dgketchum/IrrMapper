# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF (the "License");
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
import pickle
import pkg_resources
from datetime import datetime
from pandas import DataFrame, Series
from numpy import linspace, round, ceil, asarray
from numpy.random import shuffle, choice

from fiona import open as fopen
from rasterio import open as rasopen
from shapely.geometry import shape, Polygon, Point, mapping
from shapely.ops import unary_union

from sat_image.image import LandsatImage, Landsat5, Landsat7, Landsat8

WRS_2 = pkg_resources.resource_filename('spatial_data', 'wrs2_descending.shp')
temp_points = pkg_resources.resource_filename('pixel_prep', os.path.join('temp', 'sample_points.shp'))

'''
This script contains a class meant to gather data from rasters using a polygon shapefile.  The high-level 
function `compose_data_array` will return a numpy.ndarray object ready for a learning algorithm.  
'''


class PixelTrainingArray(object):
    def __init__(self, training_shape, images, instances):

        self.is_sampled = False
        self.has_data = False

        self.save_points_shape = None
        self.m_instances = instances
        self.extracted_points = {}
        self.data_dict = None
        self.dataframe = None

        self.object_id = None

        landsat_map = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        dirs = [os.path.join(images, x) for x in os.listdir(images) if os.path.isdir(os.path.join(images, x))]
        objs = [LandsatImage(os.path.join(images, x)).satellite for x in dirs]

        self.images = [landsat_map[x](y) for x, y in zip(objs, dirs)]
        self.landsat = self.images[0]
        self.path, self.row = self.landsat.target_wrs_path, self.landsat.target_wrs_row
        self.vectors = training_shape

        self.tile_bbox, self.tile_meta, self.tile_polys, self.tile_area = self.tile_geometry

    def extract_to_sample(self, save_points=False):
        self.save_points_shape = save_points
        self.sample_coverage()
        self.make_data_array()

    def sample_coverage(self):
        """ Create a clipped training set and inverse training set from polygon shapefiles.

        This complicated-looking function finds the wrs_2 descending Landsat tile corresponding
        to the path row provided, gets the bounding box and profile (aka meta) from
        compose_array.get_tile_geometry, clips the training data to the landsat tile, then performs a
        union to reduce the number of polygon objects.
        :param points:
        :param save_points:
        :return: None
        """

        time = datetime.now()

        union = unary_union(self.tile_polys)
        interior_rings_dissolved = []
        self.object_id = 1
        pos_instance_ct = 0
        for poly in union:
            interior_rings_dissolved.append(poly.exterior.coords)
            fractional_area = poly.area / self.tile_area
            required_points = ceil(fractional_area * self.m_instances * 0.5)
            x_range, y_range = self._random_points_array(poly.bounds)
            poly_pt_ct = 0
            for coord in zip(x_range, y_range):
                if poly_pt_ct < required_points:
                    if Point(coord[0], coord[1]).within(poly):
                        self._add_entry(coord, val=1)
                        poly_pt_ct += 1
                        pos_instance_ct += 1
                else:
                    break
        if pos_instance_ct > self.m_instances * 0.5:
            over = round(pos_instance_ct - self.m_instances * 5)
            keys = asarray(list(self.extracted_points.keys()))
            cut = choice(keys, over, replace=False)
            for key in cut:
                del self.extracted_points[key]

        shell = self.tile_bbox['coordinates'][0]
        inverse_polygon = Polygon(shell=shell, holes=interior_rings_dissolved)
        inverse_polygon = inverse_polygon.buffer(0)
        inverse_polygon = unary_union(inverse_polygon)
        coords = inverse_polygon.bounds
        x_range, y_range = self._random_points_array(coords)
        required_points = round(self.m_instances * 0.5)
        count = 0
        time = datetime.now()
        for coord in zip(x_range, y_range):
            if count < required_points:
                if Point(coord[0], coord[1]).within(inverse_polygon):
                    self._add_entry(coord, val=0)
                    count += 1
                    if count % 100 == 0:
                        print('Count {} of {} negative instances'
                              ' in {} seconds'.format(count, required_points,
                                                      (datetime.now() - time).seconds))
            else:
                break

        print('Total area in decimal degrees: {}\n'
              'Area irrigated: {}\n'
              'Fraction irrigated: {}'.format(shape(self.tile_bbox).area, self.tile_area,
                                              self.tile_area / shape(self.tile_bbox).area))

        print('Sample operation on {} points in {} seconds'.format(self.m_instances,
                                                                   (datetime.now() - time).seconds))
        self.is_sampled = True

    def make_data_array(self):

        self.dataframe = DataFrame(data=self.extracted_points, index=self.extracted_points.keys())

        for sat_image in self.images:
            for band, path in sat_image.tif_dict.items():
                band_series = self._point_raster_extract(path, self.dataframe)
                self.dataframe = self.dataframe.join(band_series, how='outer')

        target_series = Series(self.dataframe.LTYPE)
        target_values = target_series.values

        data = {'features': self.dataframe.columns.values, 'data': self.dataframe.values,
                'target_values': target_values}

        if pickle_path:
            with open(pickle_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.has_data = True

    def save_sample_points(self, point_collection):

        points_schema = {'properties': dict(
            [('OBJECTID', 'int:10'), ('POINT_TYPE', 'int:10')]),
            'geometry': 'Point'}
        meta = self.tile_meta.copy()
        meta['schema'] = points_schema

        with fopen(temp_points, 'w', **meta) as output:
            for key, val in point_collection.items():
                props = dict([('OBJECTID', key), ('POINT_TYPE', val['POINT_TYPE'])])
                pt = Point(val['COORDS'][0], val['COORDS'][1])
                output.write({'properties': props,
                              'geometry': mapping(pt)})

    def _point_raster_extract(self, raster):

        basename = os.path.basename(raster)
        name_split = basename.split(sep='_')
        band = name_split[7].split(sep='.')[0]
        date_string = name_split[3]
        column_name = '{}_{}'.format(date_string, band)
        print('Extracting {}'.format(column_name))

        with rasopen(raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            affine = rsrc.affine

        s = Series(index=range(0, self.dataframe.shape[0]), name=column_name)
        for ind, row in self.dataframe.iterrows():
            x, y = row['X'], row['Y']
            c, r = ~affine * (x, y)
            try:
                raster_val = rass_arr[int(r), int(c)]
                s[ind] = float(raster_val)
            except IndexError:
                s[ind] = None

        return s

    @property
    def tile_geometry(self):
        with fopen(WRS_2, 'r') as wrs:
            wrs_meta = wrs.meta.copy()
            for feature in wrs:
                fp = feature['properties']
                if fp['PATH'] == self.path and fp['ROW'] == self.row:
                    bbox = feature['geometry']

        with fopen(self.vectors, 'r') as src:
            clipped = src.filter(mask=bbox)
            polys = []
            total_area = 0.
            for feat in clipped:
                geo = shape(feat['geometry'])
                polys.append(geo)
                total_area += geo.area
        return bbox, wrs_meta, polys, total_area

    @staticmethod
    def _recursive_file_gen(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                yield os.path.join(root, file)

    def _random_points_array(self, coords):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * self.m_instances)
        y_range = linspace(min_y, max_y, num=2 * self.m_instances)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points[self.object_id] = {'OBJECTID': self.object_id,
                                                 'COORDS': coord,
                                                 'X': coord[0],
                                                 'Y': coord[1],
                                                 'POINT_TYPE': val}
        self.object_id += 1


if __name__ == '__main__':
    home = os.path.expanduser('~')
    image = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'landsat_data')
    train_shape = pkg_resources.resource_filename('spatial_data',
                                                  os.path.join('MT', 'FLU_2017_Irrig.shp'))
    PixelTrainingArray(training_shape=train_shape, images=image, instances=100).extract_to_sample()

# ========================= EOF ====================================================================

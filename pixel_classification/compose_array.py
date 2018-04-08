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
import pickle
import pkg_resources
from datetime import datetime
from warnings import warn
from pandas import DataFrame, Series
from numpy import linspace, round, max, nan, unique, cumsum
from numpy.random import shuffle
from sklearn.decomposition import PCA

from fiona import open as fopen
from rasterio import open as rasopen
from shapely.geometry import shape, Polygon, Point, mapping
from shapely.ops import unary_union
from pyproj import Proj, transform

from sat_image.image import LandsatImage, Landsat5, Landsat7, Landsat8
from pixel_classification.band_map import band_map

WRS_2 = pkg_resources.resource_filename('spatial_data', 'wrs2_descending.shp')

'''
This script contains a class meant to gather data from rasters using a polygon shapefile.  The high-level 
method `extract_sample` will return a numpy.ndarray object ready for a learning algorithm.  
'''


class PixelTrainingArray(object):
    def __init__(self, training_shape, images, instances):

        self.image_directory = images

        self.is_sampled = False
        self.has_data = False
        self.is_binary = None

        self.features = None
        self.data = None
        self.target_values = None

        self.m_instances = instances
        self.extracted_points = DataFrame(columns=['OBJECTID', 'X', 'Y', 'POINT_TYPE'])

        self.object_id = None

        landsat_map = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        dirs = [os.path.join(images, x) for x in os.listdir(images) if os.path.isdir(os.path.join(images, x))]
        objs = [LandsatImage(os.path.join(images, x)).satellite for x in dirs]

        self.band_map = band_map()

        self.images = [landsat_map[x](y) for x, y in zip(objs, dirs)]
        self.current_img = self.images[0]
        self.path, self.row = self.current_img.target_wrs_path, self.current_img.target_wrs_row
        self.vectors = training_shape
        self.coord_system = self.current_img.rasterio_geometry['crs']

    def extract_sample(self, save_points=False):
        self.sample_coverage()
        self.make_data_array()
        if save_points:
            self.save_sample_points()

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

        union = unary_union(self.polygons)
        positive_area = sum([x.area for x in self.polygons])
        interior_rings_dissolved = []
        self.object_id = 0
        pos_instance_ct = 0
        for poly in union:
            interior_rings_dissolved.append(poly.exterior.coords)
            fractional_area = poly.area / positive_area
            required_points = max([1, fractional_area * self.m_instances * 0.5])
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

        self.extracted_points.infer_objects()
        print('Total area in decimal degrees: {}\n'
              'Area irrigated: {}\n'
              'Fraction irrigated: {}'.format(shape(self.tile_bbox).area, positive_area,
                                              positive_area / shape(self.tile_bbox).area))
        print('Requested {} instances, random point placement resulted in {}'.format(self.m_instances,
                                                                                     len(self.extracted_points)))
        print('Sample operation completed in {} seconds'.format(self.m_instances,
                                                                (datetime.now() - time).seconds))
        self.is_sampled = True

    def make_data_array(self):

        for sat_image in self.images:
            self.current_img = sat_image
            for band, path in sat_image.tif_dict.items():
                if band.replace('b', '') in self.band_map[sat_image.satellite]:
                    band_series = self._point_raster_extract(path)
                    self.extracted_points = self.extracted_points.join(band_series,
                                                                       how='outer')
            for path in sat_image.masks:
                mask_series = self._point_raster_extract(path)
                self.extracted_points = self.extracted_points.join(mask_series,
                                                                   how='outer')

        data_array, targets = self._purge_array()

        data = {'features': data_array.columns.values,
                'data': data_array.values,
                'target_values': targets}

        for key, val in data.items():
            setattr(self, key, val)

        with open(self.data_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._check_targets(targets)
        self.has_data = True

    def principal_components(self, return_percentile=None, n_components=None):
        """ Extract eigenvectors and eigenvalue, return desired PCAs""
        :return:
        """

        if not self.has_data:
            warn('There is no data to perform PCA on.  Run make_data_array.')
            return None

        pca = None

        if n_components:
            pca = PCA(n_components=n_components, copy=True, whiten=False)
            pca.fit(self.data)
            self.data = pca.transform(self.data)
        elif return_percentile:
            pca = PCA(return_percentile, copy=True, whiten=False)
            pca.fit(self.data)
            self.data = pca.transform(self.data)
            print('Cumulative sum principal components: {}\n '
                  '{} features \n {}'"%"' explained variance'.format(cumsum(pca.explained_variance_ratio_),
                                                                     pca.n_components_,
                                                                     pca.n_components * 100))
        return pca

    def make_binary(self, binary_true, inplace=False):
        """ Use a key value that will equate to True (1), all others to False (0)."""
        """
        :param binary_true: 
        :return: 
        """
        pass
        # if inplace:
        #     self.y[self.y_strs == binary_true] = 1
        #     self.y[self.y_strs != binary_true] = 0
        #     self.y_strs[self.y_strs != binary_true] = '{}{}'.format('N', binary_true)
        #     unique, _ = np.unique(self.y_strs, return_inverse=True)
        #     self.classes = unique
        #     self.class_counts = {x: list(self.y_strs).count(x) for x in self.classes}
        #     self.one_hot = get_dummies(self.y).values
        # else:
        #     new = copy.deepcopy(self)
        #     self.make_binary(binary_true, inplace=True)
        #     return new

    def _purge_array(self):

        masks = [x for x in self.extracted_points.columns.tolist() if x.endswith('mask')]
        bands = [x for x in self.extracted_points.columns.tolist() if not x.endswith('mask')]
        xp = self.extracted_points
        for m in masks:
            xp[xp[m] == 1.] = nan

        for b in bands:
            xp[xp[b] == 0.] = nan

        data_array = self.extracted_points.drop(['X', 'Y', 'OBJECTID'],
                                                axis=1, inplace=False)
        data_array.dropna(axis=0, inplace=True)

        data_array.drop(masks, axis=1, inplace=True)

        target_vals = data_array.POINT_TYPE.values

        data_array = data_array.drop(['POINT_TYPE'],
                                     axis=1, inplace=False)

        return data_array, target_vals

    def _check_targets(self, target_vals):

        unique_targets = len(unique(target_vals))

        if unique_targets == 2:
            self.is_binary = True
        elif unique_targets < 2:
            warn('This dataset has fewer than two target classes,'
                 'classification is meaningless.')
        elif unique_targets > 2:
            print('This dataset has {} unique target classes'.format(unique_targets))
            self.is_binary = False
        else:
            warn('This dataset has {} target classes'.format(unique_targets))
            self.is_binary = False

    def save_sample_points(self):

        points_schema = {'properties': dict(
            [('OBJECTID', 'int:10'), ('POINT_TYPE', 'int:10')]),
            'geometry': 'Point'}
        meta = self.tile_geometry.copy()
        meta['schema'] = points_schema

        with fopen(self.shapefile_path, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('OBJECTID', row['OBJECTID']), ('POINT_TYPE', row['POINT_TYPE'])])
                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})

        return None

    def _point_raster_extract(self, raster):

        basename = os.path.basename(raster)
        name_split = basename.split(sep='_')

        try:
            band = name_split[7].split(sep='.')[0]
            date_string = name_split[3]
        except IndexError:
            band = basename.replace('.tif', '')
            date_string = self.current_img.date_acquired_str

        column_name = '{}_{}_{}'.format(self.current_img.satellite, date_string, band)
        print('Extracting {}'.format(column_name))

        with rasopen(raster, 'r') as rsrc:
            rass_arr = rsrc.read()
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            affine = rsrc.transform

        s = Series(index=range(0, self.extracted_points.shape[0]), name=column_name)
        for ind, row in self.extracted_points.iterrows():
            x, y = self._geo_point_to_projected_coords(row['X'], row['Y'])
            c, r = ~affine * (x, y)
            try:
                raster_val = rass_arr[int(r), int(c)]
                s[ind] = float(raster_val)
            except IndexError:
                s[ind] = None

        return s

    def _random_points_array(self, coords):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * self.m_instances)
        y_range = linspace(min_y, max_y, num=2 * self.m_instances)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points = self.extracted_points.append({'OBJECTID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val}, ignore_index=True)
        self.object_id += 1

    def _geo_point_to_projected_coords(self, x, y):

        in_crs = Proj(init='epsg:4326')
        out_crs = Proj(init=self.coord_system['init'])
        x, y = transform(in_crs, out_crs, x, y)
        return x, y

    @staticmethod
    def _recursive_file_gen(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                yield os.path.join(root, file)

    @property
    def data_path(self):
        return os.path.join(self.image_directory, 'data.pkl')

    @property
    def shapefile_path(self):
        return os.path.join(self.image_directory, 'sample_points.shp')

    @property
    def model_path(self):
        return os.path.join(self.image_directory, 'model.pkl')

    @property
    def polygons(self):
        with fopen(self.vectors, 'r') as src:
            clipped = src.filter(mask=self.tile_bbox)
            polys = []
            for feat in clipped:
                geo = shape(feat['geometry'])
                polys.append(geo)

        return polys

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
                if fp['PATH'] == self.path and fp['ROW'] == self.row:
                    bbox = feature['geometry']
                    return bbox


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ====================================================================

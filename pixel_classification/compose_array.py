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
from copy import deepcopy
from datetime import datetime
from warnings import warn

from fiona import open as fopen
from numpy import linspace, max, nan, unique, cumsum
from numpy.random import shuffle
from pandas import DataFrame, Series
from pyproj import Proj, transform
from rasterio import open as rasopen
from shapely.geometry import shape, Polygon, Point, mapping
from shapely.ops import unary_union
from sklearn.decomposition import PCA

from sat_image.band_map import BandMap
from sat_image.image import LandsatImage, Landsat5, Landsat7, Landsat8

loc = os.path.dirname(__file__)
WRS_2 = loc.replace('pixel_classification',
                    os.path.join('spatial_data', 'wrs2_descending.shp'))

'''
This script contains a class meant to gather data from rasters using a polygon shapefile.  
The high-level method `extract_sample` will return a numpy.ndarray object ready for a 
learning algorithm.  
'''


class NoCoordinateReferenceError(Exception):
    pass


class UnexpectedCoordinateReferenceSystemError(Exception):
    pass


class ExcessiveCloudsError(Exception):
    pass


class PixelTrainingArray(object):
    """
    Notes: The training shape must be un-projected, in the WGS84 EPSG 4326 coordinate reference system.
    For now, images are Landsat images. The 'images' parameter is for a directory of images from
    one path,row Landsat tile.
    """

    def __init__(self, root, geography, instances=None, pickle_path=None,
                 overwrite_array=False, overwrite_points=False, max_cloud=1.0, from_dict=None,
                 ancillary_rasters=None):
        """

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
        :param pickle_path: If the data exists, specify this path to instantiate a data-filled instance
        without repeating the time-consuming sampling process. (bool)
        :param overwrite_array:
        """
        g = geography
        self.root = root
        self.path_row_dir = os.path.join(self.root, str(g.path), str(g.row))
        self.year_dir = os.path.join(self.path_row_dir, str(g.year))

        if pickle_path and not overwrite_array:
            self._from_pickle(pickle_path)

        elif from_dict and not overwrite_array:
            self._from_dict(from_dict)

        elif not overwrite_array and root:
            if os.path.isfile(os.path.join(self.year, 'data.pkl')):
                self.array_exists = True
                self.overwrite_array = overwrite_array
                self.overwrite_points = overwrite_points

        else:

            self.image_directory = root
            self.array_exists = False
            self.is_sampled = False
            self.has_data = False
            self.is_binary = None
            self.overwrite_array = overwrite_array
            self.overwrite_points = overwrite_points

            self.features = None
            self.data = None
            self.target_values = None

            self.m_instances = instances
            self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE'])
            self.extract_paths = {}
            self.model_map = {}

            self.max_cloud = max_cloud
            self.water_mask = None
            self.object_id = 0

            self.band_map = BandMap()
            self.images = self._instantiate_images()
            self.current_img = self.images[0]

            self.ancillary_rasters = ancillary_rasters

            try:
                self.path, self.row = self.current_img.target_wrs_path, self.current_img.target_wrs_row
            except AttributeError:
                self.path, self.row = self.current_img.wrs_path, self.current_img.wrs_row

            self.geography = geography
            self.coord_system = self.current_img.rasterio_geometry['crs']

    def extract_sample(self, save_points=False):

        if self.array_exists and not self.overwrite_array and not self.overwrite_points:
            print(
                'The feature array has already been sampled to {}, '
                'use overwrite=True to resample '
                'this image stack'.format(self.image_directory))
            return None

        if not os.path.isfile(self.shapefile_path):
            self.create_sample_points()

        elif self.overwrite_points:
            self.create_sample_points()

        else:
            self.populate_array_from_points()

        if save_points and self.overwrite_points:
            self.save_sample_points()

        self.make_data_array()

    def populate_array_from_points(self):
        # TODO: replace with geopandas.shp_to_dataframe
        with fopen(self.shapefile_path, 'r') as src:
            for feat in src:
                coords = feat['geometry']['coordinates']
                val = feat['properties']['POINT_TYPE']
                self._add_entry(coords, val=val)

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
                x_range, y_range = self._random_points_array(poly.bounds)
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

    def create_negative_sample_points(self):
        """
        Create an inverse training set from the inverse of the polygon shapefiles.
        """

        shell = self.tile_bbox['coordinates'][0]
        inverse_polygon = Polygon(shell=shell, holes=self.interior_rings)
        inverse_polygon = inverse_polygon.buffer(0)
        inverse_polygon = unary_union(inverse_polygon)
        coords = inverse_polygon.bounds
        x_range, y_range = self._random_points_array(coords)
        count = 0
        time = datetime.now()
        for coord in zip(x_range, y_range):
            if count < self.m_instances:
                if Point(coord[0], coord[1]).within(inverse_polygon):
                    self._add_entry(coord, val=-1)
                    count += 1
                    if count % 1000 == 0:
                        print('Count {} of {} negative instances'
                              ' in {} seconds'.format(count, int(self.m_instances),
                                                      (datetime.now() - time).seconds))
            else:
                break

        self.extracted_points.infer_objects()
        self.is_sampled = True

    def make_data_array(self):

        min_cloud = 1.
        for sat_image in self.images:
            self.current_img = sat_image
            scn = self.current_img.landsat_scene_id
            try:
                for path in sat_image.masks:
                    mask_series = self._point_raster_extract(path)
                    is_cloud_mask = path.endswith('cloud_fmask.tif')
                    if is_cloud_mask:
                        fraction_masked = mask_series.sum() / len(mask_series)
                        excessive_clouds = fraction_masked > self.max_cloud
                        if fraction_masked < min_cloud:
                            min_cloud = fraction_masked
                            self.water_mask = path.replace('cloud', 'water')

                        if excessive_clouds:
                            raise ExcessiveCloudsError(
                                '{} has {:.2f}% clouds, skipping'.format(scn,
                                                                         fraction_masked * 100.))
                        else:
                            print('Extracting {}'.format(scn))

                        self.extracted_points = self.extracted_points.join(mask_series,
                                                                           how='outer')

                for band, path in sat_image.tif_dict.items():
                    if band.replace('b', '') in self.band_map.selected[sat_image.satellite]:
                        band_series = self._point_raster_extract(path)
                        self.extracted_points = self.extracted_points.join(band_series,
                                                                           how='outer')

            except Exception:
                pass

        if self.ancillary_rasters:
            for ras in self.ancillary_rasters:
                band_series = self._point_raster_extract(ras, image=False)
                self.extracted_points = self.extracted_points.join(band_series,
                                                                   how='outer')

        data_array, targets = self._purge_array()

        for key, val in self.extract_paths.items():
            if key in data_array.columns.values:
                self.model_map[key] = val

        data = {'features': data_array.columns.values,
                'data': data_array.values,
                'target_values': targets,
                'model_map': self.model_map,
                'water_mask': self.water_mask}
        print('feature dimensions: {}'.format(data_array.shape))
        for key, val in data.items():
            setattr(self, key, val)

        with open(self.data_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=0)

        self._check_targets(targets)
        self.has_data = True

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

    def _principal_components(self, return_percentile=None, n_components=None):
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
                  '{} features \n {}'"%"' explained variance'.format(
                cumsum(pca.explained_variance_ratio_),
                pca.n_components_,
                pca.n_components * 100))
        return pca

    def _from_pickle(self, path):
        pkl = pickle.load(open(path, 'rb'))
        for key, val in pkl.items():
            setattr(self, key, val)

        self._check_targets(self.target_values)
        self.has_data = True

    def _from_dict(self, data):

        for key, val in data.items():
            setattr(self, key, val)

        self._check_targets(self.target_values)
        self.has_data = True

    def _purge_array(self):

        data_array = deepcopy(self.extracted_points)

        target_vals = Series(data_array.POINT_TYPE.values, name='POINT_TYPE')

        data_array.drop(['X', 'Y', 'FID', 'POINT_TYPE'], axis=1, inplace=True)
        masks = [x for x in data_array.columns.tolist() if x.endswith('mask')]
        bands = [x for x in data_array.columns.tolist() if not x.endswith('mask')]

        for msk in masks:
            data_array[data_array[msk] == 1.] = nan

        for bnd in bands:
            data_array[data_array[bnd] == 0.] = nan

        data_array = data_array.join(target_vals,
                                     how='outer')

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
            # print('This dataset has {} unique target classes'.format(unique_targets))
            self.is_binary = False
        else:
            warn('This dataset has {} target classes'.format(unique_targets))
            self.is_binary = False

    def _point_raster_extract(self, raster, image=True):

        if image:
            basename = os.path.basename(raster)
            name_split = basename.split(sep='_')

            try:
                band = name_split[7].split(sep='.')[0]
                date_string = name_split[3]
            except IndexError:
                band = basename.replace('.tif', '')
                date_string = self.current_img.date_acquired_str

            column_name = '{}_{:03d}{:03d}_{}_{}'.format(self.current_img.satellite,
                                                         self.path,
                                                         self.row, date_string, band)

        else:
            basename = os.path.basename(raster)
            band = basename.replace('.tif', '')
            column_name = os.path.basename(band)

        self.extract_paths[column_name] = raster

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

        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val},
                                                             ignore_index=True)
        self.object_id += 1

    def _geo_point_to_projected_coords(self, x, y):

        in_crs = Proj(init='epsg:4326')
        out_crs = Proj(init=self.coord_system['init'])
        x, y = transform(in_crs, out_crs, x, y)
        return x, y

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

        # print('Found {} bad (e.g., zero-area) geometries'.format(bad_geo_count))
        # print('Found {} valid geometries'.format(len(polys)))

        return polys

    def _instantiate_images(self):
        _dir = self.image_directory
        landsat_map = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        dirs = [x[0] for x in os.walk(_dir) if
                os.path.basename(x[0])[:3] in landsat_map.keys()]
        objs = [LandsatImage(x).satellite for x in dirs]
        image_objs = [landsat_map[x](y) for x, y in zip(objs, dirs)]
        return image_objs

    @property
    def data_path(self):
        if os.path.isfile(os.path.join(self.year_dir, 'data.pkl')):
            if not self.overwrite_array:
                return None
            else:
                os.remove(os.path.join(self.year_dir, 'data.pkl'))
                return os.path.join(self.year_dir, 'data.pkl')
        else:
            return os.path.join(self.year_dir, 'data.pkl')

    @property
    def shapefile_path(self):
        return os.path.join(self.path_row_dir, 'sample_points.shp')

    @property
    def model_path(self):
        return os.path.join(self.year_dir, 'model.pkl')

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
    pass

# ========================= EOF ====================================================================

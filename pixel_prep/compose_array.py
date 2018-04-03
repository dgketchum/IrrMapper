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
from numpy import linspace, round, min, max
from numpy.random import shuffle

from fiona import open as fopen
from rasterio import open as rasopen
from shapely.geometry import shape, Polygon, Point, mapping
from shapely.ops import unary_union

from pixel_prep.nlcd_map import map_nlcd_to_flu, nlcd_value
from sat_image.image import LandsatImage, Landsat5, Landsat7, Landsat8

WRS_2 = pkg_resources.resource_filename('spatial_data', 'wrs2_descending.shp')
temp_points = pkg_resources.resource_filename('pixel_prep', os.path.join('temp', 'sample_points.shp'))

'''
This script contains functions meant to gather data from rasters using a polygon shapefile.  The high-level 
function `compose_data_array` will return a numpy.ndarray object ready for a learning algorithm.  
'''


class PixelTrainingArray(object):
    def __init__(self, training_shape, images, **kwargs):
        mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        dirs = [os.path.join(images, x) for x in os.listdir(images) if os.path.isdir(os.path.join(images, x))]
        objs = [LandsatImage(os.path.join(images, x)).satellite for x in dirs]
        self.images = [mapping[x](y) for x, y in zip(objs, dirs)]

        self.landsat = self.images[0]
        self.vectors = training_shape

    def sample_coverage(self, points=10000, save_points=False):
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
        bbox, meta = get_tile_geometry(self.path, self.row)

        with fopen(self.training_shape, 'r') as src:
            clipped = src.filter(mask=bbox)
            polys = []
            total_area = 0.
            for feat in clipped:
                geo = shape(feat['geometry'])
                polys.append(geo)
                total_area += geo.area

        union = unary_union(polys)
        point_collection = {}
        interior_rings_dissolved = []
        obj_id = 1
        for poly in union:
            interior_rings_dissolved.append(poly.exterior.coords)
            fractional_area = poly.area / total_area
            required_points = round(fractional_area * points * 0.5)
            min_x, max_x = min(poly.bounds[0]), max(poly.bounds[2])
            min_y, max_y = min(poly.bounds[1]), max(poly.bounds[3])
            x_range = linspace(min_x, max_x, num=100)
            y_range = linspace(min_y, max_y, num=100)
            shuffle(x_range), shuffle(y_range)
            count = 0
            for coord in zip(x_range, y_range):
                if count < required_points:
                    if Point(coord[0], coord[1]).within(poly):
                        point_collection[obj_id] = {}
                        point_collection[obj_id]['OBJECTID'] = obj_id
                        point_collection[obj_id]['COORDS'] = coord
                        point_collection[obj_id]['POINT_TYPE'] = 1
                        count += 1
                        obj_id += 1
                else:
                    break

        shell = bbox['coordinates'][0]
        inverse_polygon = Polygon(shell=shell, holes=interior_rings_dissolved)
        inverse_polygon = inverse_polygon.buffer(0)
        inverse_polygon = unary_union(inverse_polygon)
        coords = inverse_polygon.bounds
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * points)
        y_range = linspace(min_y, max_y, num=2 * points)
        shuffle(x_range), shuffle(y_range)
        required_points = round(points * 0.5)
        count = 0
        time = datetime.now()
        for coord in zip(x_range, y_range):
            if count < required_points:
                if Point(coord[0], coord[1]).within(inverse_polygon):
                    point_collection[obj_id] = {}
                    point_collection[obj_id]['OBJECTID'] = obj_id
                    point_collection[obj_id]['COORDS'] = coord
                    point_collection[obj_id]['POINT_TYPE'] = 0
                    count += 1
                    obj_id += 1
                    if count % 100 == 0:
                        print('Count {} of {} in {} seconds'.format(count, required_points,
                                                                    (datetime.now() - time).seconds))
            else:
                break

        print('Total area in decimal degrees: {}\n'
              'Area irrigated: {}\n'
              'Fraction irrigated: {}'.format(shape(bbox).area, total_area,
                                              total_area / shape(bbox).area))
        if save_points:

            points_schema = {'properties': dict(
                [('OBJECTID', 'int:10'), ('POINT_TYPE', 'int:10')]),
                'geometry': 'Point'}

            meta['schema'] = points_schema

            with fopen(temp_points, 'w', **meta) as output:
                for key, val in point_collection.items():
                    props = dict([('OBJECTID', key), ('POINT_TYPE', val['POINT_TYPE'])])
                    pt = Point(val['COORDS'][0], val['COORDS'][1])
                    output.write({'properties': props,
                                  'geometry': mapping(pt)})

        print('sample operation on {} points in {} seconds'.format(points,
                                                                   (datetime.now() - time).seconds))

    @property
    def tile_geometry(self):
        with fopen(WRS_2, 'r') as wrs:
            wrs_meta = wrs.meta.copy()
            for feature in wrs:
                fp = feature['properties']
                if fp['PATH'] == self.path and fp['ROW'] == self.row:
                    bbox = feature['geometry']
                    return bbox, wrs_meta


def make_data_array(shapefile, rasters, pickle_path=None,
                    nlcd_path=None, target_shapefiles=None, count=10000):
    """ Compose numpy.ndarray prepped for a learning algorithm.
    
    
    Keyword Arguments:
    :param count: 
    :param target_shapefiles: 
    :param nlcd_path: 
    :param pickle_path: 
    :param rasters: 
    :param shapefile: .shp file from which point locations will be taken.
    :param rasters: Single pixel_prep file path, list of file paths, or a dir, from which all
    /*.tif files will be used.
    # :param transform: i.e., 'normalize', 'scale' data of real-number (continuous) variable
    :return: numpy.ndarray
    """

    df = point_target_extract(points=shapefile, nlcd_path=nlcd_path, target_shapefile=target_shapefiles,
                              count_limit=count)

    rasters = raster_paths(rasters)
    for r in rasters:
        for b in ['3', '4', '5', '10']:
            if r.endswith('B{}.TIF'.format(b)):
                band_series = point_raster_extract(r, df)
                df = df.join(band_series, how='outer')

    target_series = Series(df.LTYPE)
    map_nlcd_to_flu(target_series)
    target_values = target_series.values
    df.drop(['X', 'Y', 'ID', 'ITYPE', 'LTYPE'], inplace=True, axis=1)

    data = {'features': df.columns.values, 'data': df.values,
            'target_values': target_values}

    if pickle_path:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def raster_paths(rasters):
    """ Return list of rasters from single pixel_prep, list of rasters, or a dir.    """
    if os.path.isfile(rasters):
        return [rasters]
    elif os.path.isfile(rasters[0]):
        return rasters
    elif os.path.isdir(rasters):
        lst = list(recursive_file_gen(rasters))
        return [x for x in lst if x.endswith('.TIF')]

    else:
        raise ValueError('Must provide a single .tif, a list of .tif files, or a dir')


def recursive_file_gen(mydir):
    for root, dirs, files in os.walk(mydir):
        for file in files:
            yield os.path.join(root, file)


def point_target_extract(points, nlcd_path, target_shapefile=None,
                         count_limit=None):
    point_data = {}
    with fopen(points, 'r') as src:
        for feature in src:
            name = feature['id']
            proj_coords = feature['geometry']['coordinates']
            point_data[name] = {'point': feature['geometry'],
                                'coords': proj_coords}
            # point_crs = src.profile['crs']['init']
    pt_ct = 0
    for pt_id, val in point_data.items():
        pt_ct += 1
        if pt_ct < count_limit:
            pt = shape(val['point'])
            with fopen(target_shapefile, 'r') as target_src:
                has_attr = False
                for t_feature in target_src:
                    polygon = t_feature['geometry']
                    if pt.within(shape(polygon)):
                        print('pt id {}, props: {}'
                              .format(pt_id, t_feature['properties']))
                        props = t_feature['properties']
                        point_data[pt_id]['properties'] = {'IType': props['IType'],
                                                           'LType': props['LType']}

                        has_attr = True
                        break

                if not has_attr:
                    if nlcd_path:
                        with rasopen(nlcd_path, 'r') as rsrc:
                            rass_arr = rsrc.read()
                            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
                            affine = rsrc.affine

                            x, y = val['coords']
                            col, row = ~affine * (x, y)
                            raster_val = rass_arr[int(row), int(col)]
                            ltype_dct = {'IType': None,
                                         'LType': str(raster_val)}
                            point_data[pt_id]['properties'] = ltype_dct
                            print('id {} has no FLU, '
                                  'nlcd {}'.format(pt_id,
                                                   nlcd_value(ltype_dct['LType'])))
                    else:
                        ltype_dct = {'IType': None,
                                     'LType': None}
                        point_data[pt_id]['properties'] = ltype_dct

    idd = []
    ltype = []
    itype = []
    x = []
    y = []
    ct = 0
    for pt_id, val in point_data.items():
        ct += 1
        if ct < count_limit:
            idd.append(pt_id)
            ltype.append(val['properties']['LType'])
            itype.append(val['properties']['IType'])
            x.append(val['coords'][0])
            y.append(val['coords'][1])
        else:
            break
    dct = dict(zip(['ID', 'LTYPE', 'ITYPE', 'X', 'Y'],
                   [idd, ltype, itype, x, y]))
    df = DataFrame(data=dct)

    return df


def point_raster_extract(raster, points):
    """ Get point values from a pixel_prep.

    :param raster: local_raster
    :param points: Shapefile of points.
    :return: Dict of coords, row/cols, and values of pixel_prep at that point.
    """

    basename = os.path.basename(raster)
    name_split = basename.split(sep='_')
    band = name_split[7].split(sep='.')[0]
    date_string = name_split[3]
    column_name = '{}_{}'.format(date_string, band)
    print('pixel_prep {}'.format(column_name))

    with rasopen(raster, 'r') as rsrc:
        rass_arr = rsrc.read()
        rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
        affine = rsrc.affine

    s = Series(index=range(0, points.shape[0]), name=column_name)
    for ind, row in points.iterrows():
        x, y = row['X'], row['Y']
        c, r = ~affine * (x, y)
        try:
            raster_val = rass_arr[int(r), int(c)]
            s[ind] = float(raster_val)
        except IndexError:
            s[ind] = None

    return s


def _point_attrs(pt_data, index):
    target = Series(name='target', index=index)
    for key, val in pt_data.items():
        target.iloc[int(key)] = pt_data[key]['land_type']
    return target


if __name__ == '__main__':
    home = os.path.expanduser('~')
    image = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'landsat_data')
    train_shape = pkg_resources.resource_filename('spatial_data', os.path.join('MT',
                                                                               'FLU_2017_Irrig.shp'))
    PixelTrainingArray(training_shape=train_shape, images=image)
# ========================= EOF ====================================================================

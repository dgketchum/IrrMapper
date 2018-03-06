# =============================================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from pandas import DataFrame, Series, Index

from fiona import open as fopen
from rasterio import open as rasopen
from shapely.geometry import shape

'''
This script contains functions meant to gather data from rasters using a points shapefile.  The high-level 
function `compose_data_array` will return a numpy.ndarray object ready for a learning algorithm.  
'''


def load_irrigation_data(shapefile, rasters, pickle_path=None,
                         nlcd_path=None, target_shapefiles=None):
    """ Compose numpy.ndarray prepped for a learning algorithm.
    
    
    Keyword Arguments:
    :param target_shapefiles: 
    :param nlcd_path: 
    :param pickle_path: 
    :param rasters: 
    :param shapefile: .shp file from which point locations will be taken.
    :param rasters: Single raster file path, list of file paths, or a dir, from which all
    /*.tif files will be used.
    # :param transform: i.e., 'normalize', 'scale' data of real-number (continuous) variable
    :return: numpy.ndarray
    """

    df = None
    target = None

    target = point_target_extract(points=shapefile, nlcd_path=nlcd_path,
                                  target_shapefile=target_shapefiles)
    df = DataFrame(target)

    rasters = raster_paths(rasters)
    for r in rasters:
        for b in ['3', '4', '5', '10']:
            if r.endswith('B{}.TIF'.format(b)):
                band_series = point_raster_extract(r, shapefile)
                df = df.join(band_series, how='outer')

    # combined = df.join(target, how='outer')
    # combined[combined == 0.] = nan
    # combined.dropna(axis=0, how='any', inplace=True)

    data = {'classes': target.unique(), 'data': df.values,
            'target_values': target.values}

    if pickle_path:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def raster_paths(rasters):
    """ Return list of rasters from single raster, list of rasters, or a dir.    """
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


def point_target_extract(points, nlcd_path=None, target_shapefile=None):
    data = Series()
    point_data = {}

    points = ([pt for pt in fopen(points)])

    with fopen(target_shapefile, 'r') as target:
        for poly in target:
            print(poly)
            for i, pt in enumerate(points):
                point = shape(pt['geometry'])
                if point.within(shape(poly['geometry'])):
                    print(i, shape(points[i]['geometry']))

    with rasopen(nlcd_path, 'r') as rsrc:
        rass_arr = rsrc.read()
        rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
        affine = rsrc.affine
        raster_crs = rsrc.profile['crs']['init']

    return data


def point_raster_extract(raster, points, get_point_attrs=False):
    """ Get point values from a raster.
    
    :param get_point_attrs: 
    :param raster: local_raster
    :param points: Shapefile of points.
    :return: Dict of coords, row/cols, and values of raster at that point.
    """
    point_data = {}

    basename = os.path.basename(raster)
    name_split = basename.split(sep='_')
    band = name_split[7].split(sep='.')[0]
    date_string = name_split[3]
    column_name = '{}_{}'.format(date_string, band)
    print('raster {}'.format(column_name))

    with fopen(points, 'r') as src:
        for feature in src:
            name = feature['id']
            proj_coords = feature['geometry']['coordinates']
            point_data[name] = {'coords': proj_coords}
            point_crs = src.profile['crs']['init']

    with rasopen(raster, 'r') as rsrc:
        rass_arr = rsrc.read()
        rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
        affine = rsrc.affine
        raster_crs = rsrc.profile['crs']['init']

    if point_crs != raster_crs:
        raise ValueError('Points and raster are not in same coordinate system.')

    index = Index(range(len(point_data)))
    point_series = Series(name=column_name, index=index)

    for key, val in point_data.items():
        x, y = val['coords']
        col, row = ~affine * (x, y)
        raster_val = rass_arr[int(row), int(col)]
        point_series.iloc[int(key)] = float(raster_val)

    if get_point_attrs:
        target_series = _point_attrs(point_data, index)
        return target_series, point_series

    return point_series


def _point_attrs(pt_data, index):
    target = Series(name='target', index=index)
    for key, val in pt_data.items():
        target.iloc[int(key)] = pt_data[key]['land_type']
    return target


if __name__ == '__main__':
    home = os.path.expanduser('~')
    montana = os.path.join(home, 'images', 'irrigation', 'MT')
    images = os.path.join(montana, 'landsat', 'LC8_39_27')
    centroids = os.path.join(montana, 'SunAreaTest', 'hex_centoids_1000m_intersect_Z12_LItype.shp')
    spatial = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial')
    p_path = os.path.join(spatial, 'pick.pickle')
    nlcd = os.path.join(montana, 'nlcd_Z12.tif')
    flu = os.path.join(montana, 'P39R27_Test', 'FLU_2017_Irrigation_Z12.shp')
    data = load_irrigation_data(centroids, images, pickle_path=p_path, nlcd_path=nlcd,
                                target_shapefiles=flu)

# ========================= EOF ====================================================================

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

from fiona import open as fopen
from pandas import DataFrame, Series
from rasterio import open as rasopen
from shapely.geometry import shape

from pixel_prep.nlcd_map import map_nlcd_to_flu, nlcd_value

# import sys
# cwd = os.getcwd()
# sys.path.append(cwd.replace('pixel_classification', 'spatial_data'))
# sys.path.append(cwd)


'''
This script contains functions meant to gather data from rasters using a points shapefile.  The high-level 
function `compose_data_array` will return a dict of numpy.ndarray objects ready for a learning algorithm.  
'''


def load_irrigation_data(shapefile, rasters, pickle_path=None,
                         nlcd_path=None, target_shapefiles=None, count=100000):
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


def point_target_extract(points, nlcd_path,
                         target_shapefile=None, count_limit=None):
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

    montana = os.path.join(home, 'pixel_prep', 'irrigation', 'MT')
    images = os.path.join(montana, 'landsat', 'LC8_39_27')
    centroids = os.path.join(montana, 'P39R27_Test', 'centroids_Z12.shp')
    nlcd = os.path.join(montana, 'nlcd_Z12.tif')
    flu = os.path.join(montana, 'P39R27_Test', 'FLU_2017_All_clip.shp')

    spatial = os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial_data')
    p_path = os.path.join(spatial, 'P39R27_Test_all.pkl')

    data = load_irrigation_data(centroids, images, pickle_path=p_path,
                                nlcd_path=nlcd, target_shapefiles=flu,
                                )

# ========================= EOF ====================================================================

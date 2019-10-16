# =============================================================================================
# Copyright 2018 dgketchum #
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at
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
from glob import glob
abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

target_bands = ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF')

def assign_shapefile_class_code_binary(shapefile):
    if 'irrigated' in shapefile and 'unirrigated' not in shapefile:
        return 1
    return 0

def assign_shapefile_class_code(shapefile):
    if 'irrigated' in shapefile and 'unirrigated' not in shapefile:
        return 0
    if 'unirrigated' in shapefile or 'wetlands' in shapefile:
        return 1
    if 'uncultivated' in shapefile:
        return 2
    if 'fallow' in shapefile:
        return 3

def assign_shapefile_year(shapefile):
    # get the shapefile name, not the whole path.
    return 2013


def landsat_rasters():
    # keys: satellite number
    # values: bands that are available for that satellite.
    b = {1: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'),
         2: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'),
         3: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'),
         4: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'),
         5: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF', 'B7.TIF'),

         7: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF',
             'B6_VCID_1.TIF', 'B6_VCID_2.TIF', 'B7.TIF', 'B8.TIF'),

         8: ('B1.TIF', 'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 'B6.TIF',
             'B7.TIF', 'B8.TIF', 'B9.TIF', 'B10.TIF', 'B11.TIF')}
    return b


def climate_rasters():
    a = ('pr.tif', 'pet.tif', 'tmmn.tif', 'tmmx.tif', 'etr.tif')
    return a


def ancillary_rasters():
    a = ('lst.tif', 'ssebop_etrf.tif')
    return a


def static_rasters():
    a = ('slope.tif', 'aspect.tif', 'elevation_diff.tif')
    return a


def mask_rasters():
    m = ('cloud_fmask.tif', )# , 'water_fmask.tif')
    return m


if __name__ == '__main__':
    pass

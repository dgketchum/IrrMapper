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
    if 'unirrigated' in shapefile:
        return 1
    if 'uncultivated' in shapefile or 'wetlands' in shapefile:
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

def irrigated_path_rows_mt():
    ls = [(37, 28), (39, 27), (42, 26), (42, 27), (41, 28), (41, 27), (41, 26), (40, 27),
            (40, 28), (39, 28), (39, 29), (38, 26), (38, 28), (34, 27), (35, 27), (36, 26), 
            (36, 27), (36, 28), (37, 26)]
    return ls

def cdl_crop_values():
    return {1: 'Corn',
            2: 'Cotton',
            3: 'Rice',
            4: 'Sorghum',
            5: 'Soybeans',
            6: 'Sunflower',
            10: 'Peanuts',
            11: 'Tobacco',
            12: 'Sweet Corn',
            13: 'Pop or Orn Corn',
            14: 'Mint',
            21: 'Barley',
            22: 'Durum Wheat',
            23: 'Spring Wheat',
            24: 'Winter Wheat',
            25: 'Other Small Grains',
            26: 'Dbl Crop WinWht / Soybeans',
            27: 'Rye',
            28: 'Oats',
            29: 'Millet',
            30: 'Speltz',
            31: 'Canola',
            32: 'Flaxseed',
            33: 'Safflower',
            34: 'Rape Seed',
            35: 'Mustard',
            36: 'Alfalfa',
            37: 'Other Hay / NonAlfalfa',
            38: 'Camelina',
            39: 'Buckwheat',
            41: 'Sugarbeets',
            42: 'Dry Beans',
            43: 'Potatoes',
            44: 'Other Crops',
            45: 'Sugarcane',
            46: 'Sweet Potatoes',
            47: 'Misc Vegs & Fruits',
            48: 'Watermelons',
            49: 'Onions',
            50: 'Cucumbers',
            51: 'Chick Peas',
            52: 'Lentils',
            53: 'Peas',
            54: 'Tomatoes',
            55: 'Caneberries',
            56: 'Hops',
            57: 'Herbs',
            58: 'Clover/Wildflowers',
            61: 'Fallow/Idle Cropland',
            66: 'Cherries',
            67: 'Peaches',
            68: 'Apples',
            69: 'Grapes',
            70: 'Christmas Trees',
            71: 'Other Tree Crops',
            72: 'Citrus',
            74: 'Pecans',
            75: 'Almonds',
            76: 'Walnuts',
            77: 'Pears',
            204: 'Pistachios',
            205: 'Triticale',
            206: 'Carrots',
            207: 'Asparagus',
            208: 'Garlic',
            209: 'Cantaloupes',
            210: 'Prunes',
            211: 'Olives',
            212: 'Oranges',
            213: 'Honeydew Melons',
            214: 'Broccoli',
            216: 'Peppers',
            217: 'Pomegranates',
            218: 'Nectarines',
            219: 'Greens',
            220: 'Plums',
            221: 'Strawberries',
            222: 'Squash',
            223: 'Apricots',
            224: 'Vetch',
            225: 'Dbl Crop WinWht/Corn',
            226: 'Dbl Crop Oats/Corn',
            227: 'Lettuce',
            229: 'Pumpkins',
            230: 'Dbl Crop Lettuce/Durum Wht',
            231: 'Dbl Crop Lettuce/Cantaloupe',
            232: 'Dbl Crop Lettuce/Cotton',
            233: 'Dbl Crop Lettuce/Barley',
            234: 'Dbl Crop Durum Wht/Sorghum',
            235: 'Dbl Crop Barley/Sorghum',
            236: 'Dbl Crop WinWht/Sorghum',
            237: 'Dbl Crop Barley/Corn',
            238: 'Dbl Crop WinWht/Cotton',
            239: 'Dbl Crop Soybeans/Cotton',
            240: 'Dbl Crop Soybeans/Oats',
            241: 'Dbl Crop Corn/Soybeans',
            242: 'Blueberries',
            243: 'Cabbage',
            244: 'Cauliflower',
            245: 'Celery',
            246: 'Radishes',
            247: 'Turnips',
            248: 'Eggplants',
            249: 'Gourds',
            250: 'Cranberries',
            254: 'Dbl Crop Barley/Soybeans'}


def cdl_non_crop_values():
    return {37: 'Other Hay/Non Alfalfa',
            59: 'Sod/Grass Seed',
            60: 'Switchgrass',
            63: 'Forest',
            64: 'Shrubland',
            65: 'Barren',
            81: 'Clouds/No Data',
            82: 'Developed',
            83: 'Water',
            87: 'Wetlands',
            88: 'Nonag/Undefined',
            92: 'Aquaculture',
            111: 'Open Water',
            112: 'Perennial Ice/Snow',
            121: 'Developed/Open Space',
            122: 'Developed/Low Intensity',
            123: 'Developed/Med Intensity',
            124: 'Developed/High Intensity',
            131: 'Barren',
            141: 'Deciduous Forest',
            142: 'Evergreen Forest',
            143: 'Mixed Forest',
            152: 'Shrubland',
            176: 'Grass/Pasture',
            190: 'Woody Wetlands',
            195: 'Herbaceous Wetlands'}


if __name__ == '__main__':
    pass

# =============================================================================================
# Copyright 2018 dgketchum
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
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)


def landsat_rasters():
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


def ancillary_rasters():
    a = ('lst.tif', 'ssebop_etrf.tif')
    return a


def static_rasters():
    a = ('slope.tif', 'aspect.tif', 'elevation_diff.tif')
    return a


def mask_rasters():
    m = ('cloud_fmask.tif', 'water_fmask.tif')
    return m


class TrainingAssignments(object):
    def __init__(self, root):
        self.attribute_list = ['forest', 'fallow', 'irrigated', 'other']

        self.root = root
        self.shapes = None

        self.attributes = {0: {'ltype': 'irrigated', 'path': None},

                           1: {'ltype': 'dryland', 'path': None},}

                           # 2: {'ltype': 'forrest', 'path': None},
                           #
                           # 3: {'ltype': 'other', 'path': None}}

    def add_paths(self):
        for key, vector in enumerate(self.shapes):
            self.attributes[key]['path'] = os.path.join(self.root, vector)


class Idaho(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['ID_2011_Irrigated_WGS84_4030.shp', 'non_irrigated_ESPA_2011_100_200_ac.shp',
                       'ID_Public_forest_4030.shp', 'ID_Public_other_4030.shp']
        self.add_paths()

        self.path = 40
        self.row = 30
        self.year = 2011
        self.sat = 5


class Montana(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['MT_Huntley_Main_2013_3728.shp', 'MT_FLU_2017_Fallow_3728.shp',]
                       # 'MT_FLU_2017_Forrest_3728.shp', 'MT_other_3728.shp']
        self.add_paths()

        self.path = 37
        self.row = 28
        self.year = 2013
        self.sat = 8


class Nevada(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['2015_IRR_ACRE_WGS84.shp', 'NV_fallow.shp',]
                       # 'NV_forest.shp', 'NV_other.shp']
        self.add_paths()

        self.path = 41
        self.row = 32
        self.year = 2015
        self.sat = 8


class Oregon(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['harney_irrigated_2016.shp', 'harney_fallow_2016.shp',]
                       # 'OR_forest.shp', 'OR_other.shp']
        self.add_paths()

        self.path = 43
        self.row = 30
        self.year = 2016
        self.sat = 8


class Utah(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['UT_Irrigated_3732_2014.shp', 'UT_UnirrigatedAg_3732.shp',]
                       # 'UT_forest.shp', 'UT_other.shp']
        self.add_paths()

        self.path = 37
        self.row = 32
        self.year = 2014
        self.sat = 8


class Washington(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['WA_2017_irrigated_4427.shp', 'WA_2017_unirrigated_ag_4427.shp',]
                       # 'WA_Forest_Practices_Applications_4427.shp', 'WA_other_4427.shp']
        self.add_paths()

        self.path = 44
        self.row = 27
        self.year = 2017
        self.sat = 8


class MontanaTest(Montana):
    def __init__(self):
        Montana.__init__(self)

        for code, _dict in self.attributes.items():
            _dict['path'] = _dict['path'].replace(os.path.join('spatial_data', 'MT'),
                                                  os.path.join('tests', 'data', 'pixel_extract_test',
                                                               ))
            _dict['path'] = _dict['path'].replace('.shp', '_clip.shp')

        self.unique_classes = len(self.attributes.keys())

        self.sample_negative = False


if __name__ == '__main__':
    pass

# ========================= EOF ================================================================

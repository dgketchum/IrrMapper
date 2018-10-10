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

                           1: {'ltype': 'dryland', 'path': None},

                           2: {'ltype': 'forrest', 'path': None},

                           3: {'ltype': 'other', 'path': None}}

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


class MontanaHuntley(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['MT_Huntley_Main_2013_3728.shp', 'MT_FLU_2017_Fallow_3728.shp',
                       'MT_FLU_2017_Forrest_3728.shp', 'MT_other_3728.shp']
        self.add_paths()

        self.path = 37
        self.row = 28
        self.year = [2008, 2009, 2010, 2011, 2012, 2013]
        self.sat = 8


class MontanaSun(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['MT_Sun_River_2013_3927.shp', 'MT_FLU_2017_Fallow_3927.shp',
                       'MT_FLU_2017_Forrest_3927.shp', 'MT_other_3927.shp']
        self.add_paths()

        self.path = 39
        self.row = 27
        self.year = [2008, 2009, 2010, 2011, 2012, 2013]
        self.sat = 8


class Nevada(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['2015_IRR_ACRE_WGS84.shp', 'NV_fallow.shp',
                       'NV_forest.shp', 'NV_other.shp']
        self.add_paths()

        self.path = 41
        self.row = 32
        self.year = [2015]
        self.sat = 8


class Oregon(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['harney_irrigated_2016.shp', 'harney_fallow_2016.shp',
                       'OR_forest.shp', 'OR_other.shp']
        self.add_paths()

        self.path = 43
        self.row = 30
        self.year = [2016]
        self.sat = 8


class Utah(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['UT_Irrigated_3732_2014.shp', 'UT_UnirrigatedAg_3732.shp',
                       'UT_forest.shp', 'UT_other.shp']
        self.add_paths()

        self.path = 37
        self.row = 32
        self.year = [2014]
        self.sat = 8


class Washington(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        self.shapes = ['WA_2017_irrigated_4427.shp', 'WA_2017_unirrigated_ag_4427.shp',
                       'WA_Forest_Practices_Applications_4427.shp', 'WA_other_4427.shp']
        self.add_paths()

        self.path = 44
        self.row = 27
        self.year = [2017]
        self.sat = 8


# class MontanaTest(MontanaSun):
#     def __init__(self):
#         MontanaHuntley.__init__(self)
#
#         for code, _dict in self.attributes.items():
#             _dict['path'] = _dict['path'].replace(os.path.join('spatial_data', 'MTH'),
#                                                   os.path.join('tests', 'data', 'pixel_extract_test',
#                                                                ))
#             _dict['path'] = _dict['path'].replace('.shp', '_clip.shp')
#
#         self.unique_classes = len(self.attributes.keys())
#
#         self.sample_negative = False


SELECTED_LIST = [(34, 34),
                 (36, 30),
                 (41, 30)]

TARGET_LIST = [(31, 27),
               (31, 28),
               (31, 26),
               (35, 27),
               (31, 31),
               (35, 28),
               (31, 32),
               (31, 29),
               (35, 26),
               (31, 30),
               (35, 31),
               (31, 35),
               (35, 32),
               (31, 36),
               (35, 29),
               (31, 33),
               (35, 30),
               (31, 34),
               (35, 35),
               (35, 36),
               (35, 33),
               (31, 37),
               (35, 34),
               (31, 38),
               (29, 27),
               (29, 28),
               (35, 37),
               (29, 26),
               (35, 38),
               (29, 31),
               (29, 32),
               (29, 29),
               (29, 30),
               (29, 35),
               (36, 35),
               (29, 36),
               (36, 36),
               (29, 33),
               (36, 33),
               (29, 34),
               (36, 34),
               (33, 27),
               (36, 37),
               (33, 28),
               (36, 38),
               (40, 31),
               (33, 26),
               (40, 32),
               (33, 31),
               (40, 29),
               (33, 32),
               (40, 30),
               (33, 29),
               (40, 35),
               (33, 30),
               (40, 36),
               (40, 33),
               (40, 34),
               (40, 37),
               (34, 27),
               (34, 28),
               (34, 26),
               (34, 31),
               (34, 32),
               (34, 29),
               (34, 30),
               (34, 35),
               (34, 36),
               (34, 33),
               (34, 34),
               (34, 37),
               (34, 38),
               (38, 27),
               (38, 28),
               (44, 27),
               (44, 28),
               (38, 26),
               (42, 27),
               (38, 31),
               (44, 26),
               (42, 28),
               (38, 32),
               (38, 29),
               (44, 31),
               (42, 26),
               (38, 30),
               (44, 32),
               (42, 31),
               (38, 35),
               (44, 29),
               (44, 30),
               (42, 32),
               (38, 36),
               (42, 29),
               (38, 33),
               (42, 30),
               (38, 34),
               (42, 35),
               (44, 33),
               (44, 34),
               (42, 36),
               (42, 33),
               (38, 37),
               (32, 27),
               (42, 34),
               (38, 38),
               (32, 28),
               (32, 26),
               (43, 31),
               (36, 27),
               (32, 31),
               (32, 32),
               (43, 32),
               (36, 28),
               (32, 29),
               (43, 29),
               (32, 30),
               (43, 30),
               (36, 26),
               (32, 35),
               (36, 31),
               (43, 35),
               (32, 36),
               (36, 32),
               (32, 33),
               (36, 29),
               (43, 33),
               (32, 34),
               (36, 30),
               (43, 34),
               (32, 37),
               (32, 38),
               (40, 27),
               (40, 28),
               (40, 26),
               (41, 27),
               (30, 27),
               (41, 28),
               (30, 28),
               (41, 26),
               (30, 26),
               (41, 31),
               (41, 32),
               (41, 29),
               (41, 30),
               (41, 35),
               (41, 36),
               (41, 33),
               (41, 34),
               (39, 27),
               (39, 28),
               (45, 27),
               (39, 26),
               (45, 28),
               (39, 31),
               (43, 27),
               (39, 32),
               (45, 26),
               (43, 28),
               (39, 29),
               (45, 31),
               (39, 30),
               (43, 26),
               (45, 32),
               (39, 35),
               (45, 29),
               (39, 36),
               (45, 30),
               (39, 33),
               (39, 34),
               (45, 33),
               (30, 31),
               (30, 32),
               (30, 29),
               (30, 30),
               (30, 35),
               (30, 36),
               (30, 33),
               (30, 34),
               (30, 37),
               (30, 38),
               (28, 36),
               (28, 37),
               (28, 38),
               (39, 37),
               (39, 38),
               (29, 37),
               (29, 38),
               (37, 27),
               (37, 28),
               (37, 26),
               (37, 31),
               (33, 35),
               (46, 28),
               (37, 32),
               (33, 36),
               (37, 29),
               (46, 26),
               (33, 33),
               (37, 30),
               (33, 34),
               (46, 31),
               (37, 35),
               (37, 36),
               (46, 29),
               (37, 33),
               (33, 37),
               (46, 30),
               (37, 34),
               (33, 38),
               (37, 37),
               (37, 38)]


def get_selected_path_rows():
    return SELECTED_LIST


def get_path_rows():
    return TARGET_LIST


if __name__ == '__main__':
    pass

# ========================= EOF ================================================================

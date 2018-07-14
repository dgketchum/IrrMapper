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

home = os.path.expanduser('~')


def return_object(key):
    _dict = {'montana': Montana,
             'montana_test': MontanaTest}
    return _dict[key]()


class TrainingAssignments(object):
    def __init__(self, root):

        self.attribute_list = ['forest', 'fallow', 'irrigated', 'other']

        self.root = root

        self.vector = 'empty'

        self.attributes = {0: {'ltype': 'irrigated', 'path': os.path.join(self.root, self.vector)},

                           1: {'ltype': 'dryland', 'path': os.path.join(self.root, self.vector)},

                           2: {'ltype': 'forrest', 'path': os.path.join(self.root, self.vector)},

                           3: {'ltype': 'other', 'path': os.path.join(self.root, self.vector)}}


class Idaho(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        shapes = ['ID_2011_Irrigated_WGS84_4030.shp', 'non_irrigated_ESPA_2011_100_200_ac.shp',
                  'ID_Public_forest_4030.shp', 'ID_Public_other_4030.shp']

        for key, vector in enumerate(shapes):
            self.attributes[key]['path'].replace(self.vector, vector)

        self.path = 40
        self.row = 30
        self.year = 2011
        self.sat = 5


class Montana(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        shapes = ['MT_Sun_River_2013_3927.shp', 'MT_FLU_2017_Fallow_3927.shp',
                  'MT_FLU_2017_Forrest_3927.shp', 'MT_other_3927.shp']

        for key, vector in enumerate(shapes):
            self.attributes[key]['path'].replace(self.vector, vector)

        self.path = 39
        self.row = 27
        self.year = 2013
        self.sat = 8


class Nevada(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        shapes = ['2015_IRR_ACRE_NV/2015_IRR_ACRE.shp', 'NV_fallow.shp',
                  'NV_forest.shp', 'NV_other.shp']

        for key, vector in enumerate(shapes):
            self.attributes[key]['path'].replace(self.vector, vector)

        self.path = 41
        self.row = 32
        self.year = 2015
        self.sat = 8


class Nevada(TrainingAssignments):

    def __init__(self, root):
        TrainingAssignments.__init__(self, root)

        shapes = ['2015_IRR_ACRE_NV/2015_IRR_ACRE.shp', 'NV_fallow.shp',
                  'NV_forest.shp', 'NV_other.shp']

        for key, vector in enumerate(shapes):
            self.attributes[key]['path'].replace(self.vector, vector)

        self.path = 41
        self.row = 32
        self.year = 2015
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

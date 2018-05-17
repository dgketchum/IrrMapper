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
    _dict = {'montana': Montana}
    return _dict[key]()


class TrainingAssignments(object):
    def __init__(self, **selected_attributes):
        self.attribute_list = ['forest', 'fallow', 'irrigated']
        self.selected_attributes = None


class Montana(TrainingAssignments):

    def __init__(self, **selected_attributes):
        TrainingAssignments.__init__(self, **selected_attributes)

        self.attributes = {
            0: {'ltype': 'irrigated', 'path': os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial_data', 'MT',
                                                           'FLU_2017_Irrig.shp')},

            1: {'ltype': 'dryland', 'path': os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial_data', 'MT',
                                                         'FLU_2017_Fallow.shp')},

            2: {'ltype': 'forrest', 'path': os.path.join(home, 'PycharmProjects', 'IrrMapper', 'spatial_data', 'MT',
                                                         'FLU_2017_Forrest.shp')}}

        self.negative = {-1: {}}

        self.unique_classes = len(self.attributes.keys())

        self.sample_negative = True


class MontanaTest(Montana):
    def __init__(self):
        Montana.__init__(self)

        for code, _dict in self.attributes.items():
            _dict['path'].replace(os.path.join('spatial_data', 'MT'),
                                  os.path.join('tests', 'data', 'pixel_extract_test',
                                               ))
            _dict['path'].replace('.shp', '_clip.shp')


if __name__ == '__main__':
    pass

# ========================= EOF ================================================================

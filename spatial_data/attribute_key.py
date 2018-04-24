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

def montana():
    d = {1: {'path': os.path.join('PycharmProjects', 'IrrMapper', 'spatial_data', 'MT', 'FLU_2017_Irrig.shp'),
             'mode': 'normal',
             'attrs': 'all'},

         2: {'path': os.path.join('PycharmProjects', 'IrrMapper', 'spatial_data', 'MT', 'FLU_2017_Fallow.shp'),
             'mode': 'normal',
             'attrs': 'all'},

         0: {'path': os.path.join('PycharmProjects', 'IrrMapper', 'spatial_data', 'MT', 'FLU_2017_Forrest.shp')}}


if __name__ == '__main__':
    pass

# ========================= EOF ================================================================

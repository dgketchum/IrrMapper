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

NLCD_MAP = {'11': 'water',
            '12': 'ice_snow',
            '21': 'developed_open',
            '22': 'developed_low',
            '23': 'developed_med',
            '24': 'developed_high',
            '31': 'barren',
            '41': 'deciduous',
            '42': 'evergreen',
            '43': 'mixed_forest',
            '51': 'dwarf_scrub',
            '52': 'shrub',
            '71': 'grassland',
            '72': 'sedge',
            '73': 'lichen',
            '74': 'moss',
            '81': 'pasture',
            '82': 'crops',
            '90': 'woody_wetland',
            '95': 'emergent_herbaceaous'}

NLCD_MAP = {'11': '',# 'water',
            '12': '',# 'ice_snow',
            '21': '',# 'developed_open',
            '22': '',# 'developed_low',
            '23': '',# 'developed_med',
            '24': '',# 'developed_high',
            '31': '',# 'barren',
            '41': '',# 'deciduous',
            '42': '',# 'evergreen',
            '43': '',# 'mixed_forest',
            '51': '',# 'dwarf_scrub',
            '52': '',# 'shrub',
            '71': '',# 'grassland',
            '72': '',# 'sedge',
            '73': '',# 'lichen',
            '74': '',# 'moss',
            '81': '',# 'pasture',
            '82': '',# 'crops',
            '90': '',# 'woody_wetland',
            '95': '',# 'emergent_herbaceaous'}

def nlcd_key():
    return NLCD_MAP

def map_nlcd_to_flu(dataframe):

    pass

if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ====================================================================

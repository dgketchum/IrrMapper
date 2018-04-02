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
from shapely.geometry import mapping, shape
from fiona import open as fopen
from fiona import collection

WRS_2 = 'spatial_data/wrs2_descending.shp'


def clip_training_to_path_row(path, row, training_shape):
    bbox = None
    with fopen(training_shape, 'r') as src:
        clipped = src.filter(bbox=bbox)
        clipped_schema = src.schema.copy()
        with collection('clipped.shp', 'w', 'ESRI Shapefile', clipped_schema) as output:
            for elem in clipped:
                output.write({'properties': elem['properties'], 'geometry': mapping(shape(elem['geometry']))})


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================

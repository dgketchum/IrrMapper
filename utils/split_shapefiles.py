# ===============================================================================
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
# ===============================================================================
import os

import fiona

WETLAND_SHAPEFILES = ['MT_shapefile_wetlands/MT_Wetlands_West.shp',
                      'MT_shapefile_wetlands/MT_Wetlands_East.shp',
                      'AZ_shapefile_wetlands/AZ_shapefile_wetlands',
                      'CA_shapefile_wetlands/CA_Wetlands_North.shp',
                      'CA_shapefile_wetlands/CA_Wetlands_Central.shp',
                      'CA_shapefile_wetlands/CA_Wetlands_East.shp',
                      'ID_shapefile_wetlands/ID_Wetlands.shp',
                      'NM_shapefile_wetlands/NM_Wetlands.shp',
                      'NV_shapefile_wetlands/NV_Wetlands_North.shp',
                      'NV_shapefile_wetlands/NV_Wetlands_South.shp',
                      'OR_shapefile_wetlands/OR_Wetlands_East.shp',
                      'OR_shapefile_wetlands/OR_Wetlands_West.shp',
                      'UT_shapefile_wetlands/UT_Wetlands.shp',
                      'WA_shapefile_wetlands/WA_Wetlands_West.shp',
                      'WA_shapefile_wetlands/WA_Wetlands_East.shp']


def split_wetlands(out_shp, file_list):
    meta = fiona.open(file_list[0]).meta
    with fiona.open(out_shp, 'w', **meta) as output:
        for s in file_list:
            for features in fiona.open(s):
                output.write(features)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    samples = 'sample_points.shp'
    s_dir = os.path.join(home, 'PycharmProjects', 'IrrigationGIS', 'wetlands')
# ========================= EOF ====================================================================

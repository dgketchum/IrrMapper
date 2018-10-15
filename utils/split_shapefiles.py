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


def split_wetlands(in_shp):

    surface = []
    surface_parameters = ['Riverine', 'Lake', 'Freshwater Pond']
    wetland = []
    wetland_parameters = ['Freshwater Emergent Wetland', 'Freshwater Forested/Shrub Wetland']

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            if feat['properties']['WETLAND_TY'] in wetland_parameters:
                wetland.append(feat)
            if feat['properties']['WETLAND_TY'] in surface_parameters:
                surface.append(feat)

    for _type in [('open_water', surface), ('wetlands', wetland)]:
        print(_type[0])
        name = in_shp.replace('.shp', '{}.shp'.format(_type[0]))
        l = _type[1]
        with fiona.open(name, 'w', **meta) as output:
            for feat in l:
                output.write(feat)

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    s_dir = os.path.join(home, 'IrrigationGIS', 'wetlands')
    o_dir = os.path.join(home, 'IrrigationGIS', 'EE_sample', 'wetlands')
    for s in WETLAND_SHAPEFILES:
        print(s)
        split_wetlands(os.path.join(s_dir, s))

# ========================= EOF ====================================================================

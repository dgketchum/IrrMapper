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

WETLAND_SHAPEFILES = [
    # 'MT_shapefile_wetlands/MT_Wetlands_West.shp',
    # 'MT_shapefile_wetlands/MT_Wetlands_East.shp',
    # 'AZ_shapefile_wetlands/AZ_Wetlands.shp',
    # 'CA_shapefile_wetlands/CA_Wetlands_North.shp',
    # 'CA_shapefile_wetlands/CA_Wetlands_NorthCentral.shp',
    # 'CA_shapefile_wetlands/CA_Wetlands_SouthCentral.shp',
    # 'CO_shapefile_wetlands/CO_Wetlands_West.shp',
    # 'CO_shapefile_wetlands/CO_Wetlands_East.shp',
    # 'ID_shapefile_wetlands/ID_Wetlands.shp',
    # 'NM_shapefile_wetlands/NM_Wetlands.shp',
    # 'NV_shapefile_wetlands/NV_Wetlands_North.shp',
    # 'NV_shapefile_wetlands/NV_Wetlands_South.shp',
    # 'OR_shapefile_wetlands/OR_Wetlands_East.shp',
    # 'OR_shapefile_wetlands/OR_Wetlands_West.shp',
    # 'UT_shapefile_wetlands/UT_Wetlands.shp',
    # 'WA_shapefile_wetlands/WA_Wetlands_West.shp',
    'WA_shapefile_wetlands/WA_Wetlands_East.shp',
    'WY_shapefile_wetlands/WY_Wetlands_West.shp',
    'WY_shapefile_wetlands/WY_Wetlands_East.shp',
]

ID_ESPA = [('ID_1986_ESPA_WGS84.shp', 'STATUS_198'),
           ('ID_1996_ESPA_WGS84.shp', 'STATUS_199'),
           ('ID_2002_ESPA_WGS84.shp', 'STATUS_200'),
           ('ID_2006_ESPA_WGS84.shp', 'STATUS_200'),
           ('ID_2008_ESPA_WGS84.shp', 'STATUS_200'),
           ('ID_2009_ESPA_WGS84.shp', 'STATUS_200'),
           ('ID_2010_ESPA_WGS84.shp', 'STATUS_201'),
           ('ID_2011_ESPA_WGS84.shp', 'STATUS_201')]


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


def split_idaho(in_shp, prop='STATUS_201'):
    irr = []
    non_irr = []

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        for feat in src:
            if feat['properties'][prop] == 'irrigated':
                irr.append(feat)
            if feat['properties'][prop] == 'non-irrigated':
                non_irr.append(feat)

    for _type in [('irr', irr), ('non-irrigated', non_irr)]:
        print(_type[0])
        name = in_shp.replace('.shp', '_{}.shp'.format(_type[0]))
        l = _type[1]
        with fiona.open(name, 'w', **meta) as output:
            for feat in l:
                output.write(feat)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    s_dir = os.path.join(home, 'IrrigationGIS', 'training_raw', 'ID')
    o_dir = os.path.join(home, 'IrrigationGIS', 'EE_sample')
    for s in ID_ESPA:
        shp = s[0]
        p = s[1]
        file_name = os.path.join(s_dir, shp)
        split_idaho(file_name, p)

# ========================= EOF ====================================================================

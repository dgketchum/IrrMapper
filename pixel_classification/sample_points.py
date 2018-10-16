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

OPEN_WATER = ['MT_Wetlands_Eastopen_water.shp',
              'WA_Wetlands_Westopen_water.shp',
              'CA_Wetlands_NorthCentralopen_water.shp',
              'CA_Wetlands_SouthCentralopen_water.shp',
              'WY_Wetlands_Eastopen_water.shp',
              'OR_Wetlands_Eastopen_water.shp',
              'NM_Wetlandsopen_water.shp',
              'CO_Wetlands_Westopen_water.shp',
              'ID_Wetlandsopen_water.shp',
              'AZ_Wetlandsopen_water.shp',
              'CO_Wetlands_Eastopen_water.shp',
              'MT_Wetlands_Westopen_water.shp',
              'WA_Wetlands_Eastopen_water.shp',
              'NV_Wetlands_Southopen_water.shp',
              'OR_Wetlands_Westopen_water.shp',
              'CA_Wetlands_Northopen_water.shp',
              'WY_Wetlands_Westopen_water.shp',
              'UT_Wetlandsopen_water.shp',
              'NV_Wetlands_Northopen_water.shp']

MT_SHP = ['West_Bench_Canal.shp',
          'East_Fork_Main_Canal_ab_Trout_Creek.shp',
          'Broadwater_Missouri_West_Side_Canal.shp',
          'Paradise_Valley_ID.shp',
          'Broadwater_Missouri_Canal.shp',
          'Huntley_Main_Diversion.shp',
          'Vigilante_Canal.shp',
          'LYIP_North_Dakota.shp',
          'Yellowstone_Main_Diversion.shp',
          'Ruby_River.shp',
          'Eldorado.shp',
          'Glasgow_ID.shp',
          'Fort_Belknap_Main_Diversion.shp',
          'Dodson_North_Canal_Diversion.shp',
          'Sun_River_project_Below_Pishkun.shp',
          'Fort_Shaw_Canal.shp',
          'Dodson_South_Div_To_Bowdoin.shp',
          'Floweree_and_Floweree_Hamilton.shp',
          'Marshall_Canal.shp']


class PointsRunspec(object):

    def __init__(self):
        pass

    def surface_water(self):
        pass

    def wetlands(self):
        pass

    def unirrigated(self):
        pass

    def irrigated(self):
        key = {'MT': 'Montana',
               'NV': 'Nevada',
               'OR': 'Oregon',
               'UT': 'Utah',
               'WA': 'Washington'}

    def forest(self):
        pass


def sample_points(project, training, out_points, n_points):
    shp_paths = []
    for key, val in OBJECT_MAP.items():
        print('Points for {}'.format(key))

        project_state_dir = os.path.join(project, key)

        if not os.path.isdir(project_state_dir):
            try:
                os.mkdir(project_state_dir)
            except FileNotFoundError:
                os.makedirs(project_state_dir)

        geography = os.path.join(training, key)
        geo = val(geography)
        years = deepcopy(geo.year)
        for i, yr in enumerate(years):
            geo.year = yr

            geo_folder = os.path.join(project, key)
            geo_data_path = os.path.join(geo_folder, 'data.pkl')

            if not os.path.isfile(geo_data_path):
                geo_data_path = None

            p = Pta(root=geo_folder, geography=geo, instances=n_points,
                    overwrite_array=True, overwrite_points=True, pkl_path=geo_data_path)

            p.create_sample_points()
            p.save_sample_points()
            shp_paths.append(p.shapefile_path)

    fiona_merge(out_points, shp_paths)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

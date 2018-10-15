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
        pass

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

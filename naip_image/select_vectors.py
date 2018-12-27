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

from pandas import DataFrame
from geopandas import read_file


def get_shapefile(shp, **filter_attrs):
    gdf = read_file(shp)
    df = DataFrame(gdf)
    _drop = [x for x in df.columns if x not in filter_attrs['select']]
    df.drop(columns=_drop, inplace=True)
    df.sample(frac=1.)
    return df['geometry']


def visualize_geometries(geometries):
    pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tables = os.path.join(home, 'Field_Extract', 'training_tables')
    o = os.path.join(tables, 'MT_test_table.csv')
    s = os.path.join(home, 'IrrigationGIS', 'irr_gis', 'FLUS_All.shp')
    get_shapefile(s, **{'field': 'LType', 'select': ['F', 'I']})
# ========================= EOF ====================================================================

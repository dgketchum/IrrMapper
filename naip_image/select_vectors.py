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

from geopandas import read_file
from pandas import DataFrame

from naip_image. naip import ApfoNaip


def get_geometries(shp, **filter_attrs):
    gdf = read_file(shp)
    df = DataFrame(gdf)
    if filter_attrs['select']:
        _drop = [x for x in df.columns if x not in filter_attrs['select']]
        df.drop(columns=_drop, inplace=True)
    df.sample(frac=1.)
    return df['geometry']


def visualize_geometries(geometries, state='montana'):
    for g in geometries:
        center = g.centroid
        box = g.bounds
        naip = ApfoNaip(box).get_image(state)
        # TODO: bad NAIP url!!!


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extraction = os.path.join(home, 'field_extraction')

    tables = os.path.join(home, 'field_extraction', 'training_tables')
    o = os.path.join(tables, 'WA_test_table.csv')

    shapes = os.path.join(home, 'field_extraction', 'raw_shapefiles')
    s = os.path.join(shapes, 'central_WA.shp')

    geos = get_geometries(s, **{'field': 'Irrigation', 'select': []})
    visualize_geometries(geos, state='washington')
# ========================= EOF ====================================================================

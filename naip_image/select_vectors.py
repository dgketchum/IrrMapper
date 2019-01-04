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
import sys
abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
import random
import string

import matplotlib as mpl
import rasterio
import rasterio.plot
from descartes import PolygonPatch
from geopandas import read_file
from pandas import DataFrame
from shapely.geometry import Polygon

from naip_image.naip import ApfoNaip

TEMP_TIF = os.path.join(os.path.dirname(__file__), 'temp', 'temp_tile_geo.tif')


def get_geometries(shp, **filter_attrs):
    gdf = read_file(shp)
    df = DataFrame(gdf)
    if filter_attrs['select']:
        _drop = [x for x in df.columns if x not in filter_attrs['select']]
        df.drop(columns=_drop, inplace=True)
    df.sample(frac=1.)
    return df['geometry']


def get_naip_polygon(bbox):
    return Polygon([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]]])


def visualize_geometries(geometries, state='montana', out_dir=None):
    for g in geometries:
        naip_args = dict([('dst_crs', '4326'),
                          ('centroid', (g.centroid.y, g.centroid.x)),
                          ('buffer', 1000)])
        naip = ApfoNaip(**naip_args)
        array, profile = naip.get_image(state)
        naip_geometry = get_naip_polygon(naip.bbox)
        naip.save(array, profile, TEMP_TIF, crs=naip_args['dst_crs'])
        naip.close()

        src = rasterio.open(TEMP_TIF)
        rasterio.plot.show((src, 1))
        ax = mpl.pyplot.gca()
        clips = [geo for geo in geometries if geo.intersects(naip_geometry)]
        patches = [PolygonPatch(f) for f in clips]
        ax.add_collection(mpl.collections.PatchCollection(patches))

        opt = input('Keep this training data?')
        if opt in ['Yes', 'YES', 'yes', 'y']:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            _dir = os.path.join(out_dir, name)
            os.mkdir(_dir)
            os.rename(TEMP_TIF, os.path.join(_dir, '{}_multi.tif'.format(name)))

            naip_bool_name = os.path.join(_dir, '{}_bool.tif'.format(name))
            meta = src.meta.copy()
            meta.update(compress='lzw')
            with rasterio.open(naip_bool_name, 'w', **meta) as out:
                out_arr = out.read(1)
                shapes = ((geom, value) for geom, value in zip(counties.geometry, counties.LSAD_NUM))
                burned = geometries.values.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                out.write_band(1, burned)
        else:
            pass
        break


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extraction = os.path.join(home, 'field_extraction')

    tables = os.path.join(extraction, 'training_tables')
    o = os.path.join(tables, 'WA_test_table.csv')

    shapes = os.path.join(extraction, 'raw_shapefiles')
    s = os.path.join(shapes, 'central_WA.shp')

    geos = get_geometries(s, **{'field': 'Irrigation', 'select': []})
    visualize_geometries(geos, state='WA')

# ========================= EOF ====================================================================

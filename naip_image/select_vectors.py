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

from numpy import uint8
from matplotlib import pyplot as plt
from matplotlib import collections as cplt
import rasterio
import rasterio.plot
from rasterio.features import rasterize
from descartes import PolygonPatch
from geopandas import read_file
from pandas import DataFrame
from shapely.geometry import Polygon
from naip_image.naip import ApfoNaip

TEMP_TIF = os.path.join(os.path.dirname(__file__), 'temp', 'temp_tile_geo.tif')


def convert_bytes(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return num, x
        num /= 1024.0


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def get_geometries(shp, n=100):
    gdf = read_file(shp)
    gdf = gdf.head(n=n)
    df = DataFrame(gdf)
    df = df.sample(frac=1.)
    return df['geometry']


def get_naip_polygon(bbox):
    return Polygon([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]]])


def get_training_scenes(geometries, instance_label=False, state='MT', out_dir=None, year=None, n=10):

    ct = 0

    overview = os.path.join(out_dir, 'overview')
    image = os.path.join(out_dir, 'image')
    labels = os.path.join(out_dir, 'labels')

    [os.mkdir(x) for x in [overview, image, labels] if not os.path.exists(x)]

    for g in geometries:

        naip_args = dict([('dst_crs', '4326'),
                          ('centroid', (g.centroid.y, g.centroid.x)),
                          ('buffer', 1000),
                          ('year', year)])

        naip = ApfoNaip(**naip_args)
        array, profile = naip.get_image(state)
        naip.save(array, profile, TEMP_TIF)
        naip_geometry = get_naip_polygon(naip.bbox)
        src = rasterio.open(TEMP_TIF)

        vectors = [geo for geo in geometries if geo.intersects(naip_geometry)]
        fig, ax = plt.subplots()
        rasterio.plot.show((src, 1), cmap='viridis', ax=ax)

        patches = [PolygonPatch(feature, edgecolor="red", facecolor="none",
                                linewidth=1.) for feature in vectors]

        ax.add_collection(cplt.PatchCollection(patches, match_original=True))
        ax.set_xlim(naip_geometry.bounds[0], naip_geometry.bounds[2])
        ax.set_ylim(naip_geometry.bounds[1], naip_geometry.bounds[3])

        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        fig_name = os.path.join(overview, '{}.png'.format(name))
        plt.savefig(fig_name)
        plt.close(fig)

        fs, unit = file_size(fig_name)
        if fs < 300 and unit == 'KB':
            print(fs, unit)
            os.remove(fig_name)

        else:
            os.rename(TEMP_TIF, os.path.join(image, '{}.tif'.format(name)))
            naip_bool_name = os.path.join(labels, '{}.tif'.format(name))

            meta = src.meta.copy()
            meta.update(compress='lzw')
            meta.update(nodata=0)
            meta.update(count=1)

            if instance_label:
                label_values = [(f, i) for i, f in enumerate(vectors)]
            else:
                label_values = [(f, 1) for f in vectors]

            with rasterio.open(naip_bool_name, 'w', **meta) as out:
                burned = rasterize(shapes=label_values, fill=0, dtype=uint8,
                                   out_shape=(array.shape[1], array.shape[2]), transform=out.transform,
                                   all_touched=False)
                out.write(burned, 1)
            ct += 1

            plt.close()

        if ct >= n:
            break


def clean_out_training_data(parent_dir):
    views = os.path.join(parent_dir, 'overview')
    labels = os.path.join(parent_dir, 'labels')
    image = os.path.join(parent_dir, 'image')

    keep = [x[:6] for x in os.listdir(views)]
    remove = [x for x in os.listdir(labels) if x[:6] not in keep]
    [os.remove(os.path.join(labels, x)) for x in remove]
    remove = [x for x in os.listdir(image) if x[:6] not in keep]
    [os.remove(os.path.join(image, x)) for x in remove]


if __name__ == '__main__':
    home = os.path.expanduser('~')
    extraction = os.path.join(home, 'field_extraction')
    states = [('AZ', 2015), ('CA', 2018), ('CO', 2017), ('MT', 2017), ('NM', 2018),
              ('NV', 2017), ('OR', 2016), ('UT', 2018), ('WY', 2017)]
    for state, year in states[2:]:
        try:
            tables = os.path.join(extraction, 'training_data', '{}'.format(state))
            if not os.path.exists(tables):
                os.mkdir(tables)
            shape_dir = os.path.join(home, 'IrrigationGIS', 'openET', '{}'.format(state))
            shapes = os.path.join(shape_dir, '{}.shp'.format(state.lower()))
            target_number = 500
            if not os.path.exists(shapes):
                raise ValueError('{} does not exist'.format(shapes))

            geos = get_geometries(shapes, n=5 * target_number)
            get_training_scenes(geos, instance_label=True, state='{}'.format(state),
                                out_dir=tables, year=year, n=target_number)
        except Exception as e:
            print(state, e)
    # clean_out_training_data(tables)
# ========================= EOF ====================================================================

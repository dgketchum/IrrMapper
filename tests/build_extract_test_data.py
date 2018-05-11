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
import fiona
import rasterio
import subprocess
from fiona import crs
from pyproj import Proj, transform
from shapely.geometry import Polygon, mapping, shape

paths = sys.path
base = paths[7]
root = base.split('lib')[0]
RIO = os.path.join(root, 'bin', 'rio')


def build_data(coords_wsen, image_dir, training_vector, new_test_dir):
    w, s, e, n = coords_wsen
    linear_ring = [(e, n), (w, n), (w, s), (e, s)]
    schema = {'geometry': 'Polygon',
              'properties': {'FID': 'int:10'}}
    args = {'mode': 'w', 'driver': 'ESRI Shapefile', 'schema': schema, 'crs': crs.from_epsg(4326)}
    with fiona.open(os.path.join(new_test_dir, 'polygon.shp'), **args) as output:
        poly = Polygon(shell=linear_ring)
        prop = {'FID': 1}
        output.write({'geometry': mapping(poly), 'properties': prop})

    first = True
    image_dirs = [x for x in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, x))]
    for image in image_dirs:
        for tif in os.listdir(os.path.join(image_dir, image)):
            if tif.lower().endswith('.tif'):
                in_tif = os.path.join(image_dir, image, tif)
                out_tif = os.path.join(new_test_dir, image, tif)
                if first:
                    with rasterio.open(in_tif, 'r') as ras:
                        ref_system = ras.crs
                    first = False
                    init = ref_system['init']
                    proj = []
                    for tup in linear_ring:
                        proj.append(_geo_point_to_projected_coords(tup[0], tup[1], init))

                call_string = '{} clip --bounds {} {} {} {} {} {}'.format(RIO,
                                                                 str(proj[2][0]), str(proj[3][1]),
                                                                  str(proj[0][0]), str(proj[1][1]),
                                                                  in_tif, out_tif)
                print(call_string)
                subprocess.run(call_string, shell=True)
                break

    clip_train_vector = training_vector.replace('.shp', '_clip.shp')
    training_clip = os.path.join(new_test_dir, os.path.basename(clip_train_vector))

    with fiona.open(training_vector) as trn:
        clipped = trn.filter(bbox=(w, s, e, n))

        args = {'mode': 'w', 'driver': 'ESRI Shapefile',
                'schema': trn.schema, 'crs': crs.from_epsg(4326)}

        with fiona.open(training_clip, **args) as clip:
            for elem in clipped:
                clip.write(elem)


def _geo_point_to_projected_coords(x, y, out_crs):
    in_crs = Proj(init='epsg:4326')
    out_crs = Proj(init=out_crs)
    x, y = transform(in_crs, out_crs, x, y)
    return x, y


if __name__ == '__main__':
    coords = -111.67, 47.17, -111.20, 47.48
    _dir = os.path.join(os.path.dirname(__file__).replace('tests', 'landsat_data'), '39', '27', '2015')
    train = os.path.join(os.path.dirname(__file__).replace('tests', 'spatial_data'), 'MT', 'FLU.shp')
    test_dir = os.path.join(os.path.dirname(__file__), 'data', 'pixel_extract_test_2')
    build_data(coords, _dir, train, test_dir)
# ========================= EOF ====================================================================

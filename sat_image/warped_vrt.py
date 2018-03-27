# =============================================================================================
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
# =============================================================================================

from __future__ import division

import os

import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio import shutil as rio_shutil
from rasterio import open as rasopen

from sat_image.image import Landsat5, Landsat7, Landsat8


def warp_vrt(directory, sat):
    list_dir = os.listdir(directory)
    first = True

    for d in list_dir:
        if first:

            mapping = {'LC8': Landsat8, 'LE7': Landsat7, 'LT5': Landsat5}

            landsat = mapping[sat](os.path.join(directory, d))
            dst = landsat.rasterio_geometry

            vrt_options = {'resampling': Resampling.cubic,
                           'dst_crs': dst['crs'],
                           'dst_transform': dst['transform'],
                           'dst_height': dst['height'],
                           'dst_width': dst['width']}

            print('geometry: {} \n {}'.format(d, vrt_options))
            print('shape: {}'.format(landsat.shape))

            first = False

        else:
            paths = []
            band_mapping = {'LC8': ['3', '4', '5', '10'],
                            'LE7': ['2', '3', '4', '6_VCID_2']}
            for x in os.listdir(os.path.join(directory, d)):
                for y in band_mapping[sat]:
                    if x.endswith('B{}.TIF'.format(y)):
                        paths.append(os.path.join(directory, d, x))

            for path in paths:
                print(path)
                with rasopen(path, 'r+') as src:
                    src.crs = dst['crs']
                    with WarpedVRT(src, **vrt_options) as vrt:
                        data = vrt.read()

                        dst_dir, name = os.path.split(path)
                        print(dst_dir, name)
                        outfile = os.path.join(dst_dir, name.replace('_T', '_TA'))
                        rio_shutil.copy(vrt, outfile, driver='GTiff')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    images = os.path.join(home, 'landsat_images', 'LE7_39_27')
    warp_vrt(images, 'LE7')

# ========================= EOF ================================================================

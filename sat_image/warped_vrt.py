# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF (the "License");
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
from datetime import datetime

from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio import open as rasopen

from sat_image.image import Landsat5, Landsat7, Landsat8


def warp_vrt(directory, sat, delete_extras=False):
    """ Read in image geometry, resample subsequent images to same grid.

    The purpose of this function is to snap many Landsat images to one geometry. Use Landsat578
    to download and unzip them, then run them through this to get identical geometries for analysis.
    Files
    :param directory: A directory containing sub-directories of Landsat images.
    :param sat: Landsat satellite; 'LT5', 'LE7', 'LC8'
    :param delete_extras: Remove all but targeted bands and the .MTL file.
    :return: None
    """
    vrt_options = {}
    list_dir = os.listdir(directory)
    first = True

    band_mapping = {'LC8': ['3', '4', '5', '10'],
                    'LE7': ['2', '3', '4', '6_VCID_2'],
                    'LT5': ['2', '3', '4', '6']}

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
            message = """
            This directory has been resampled to same grid.
            Master grid is {}.
            {}
            """.format(d, datetime.now())
            with open(os.path.join(directory, 'resample_meta.txt'), 'w') as f:
                f.write(message)
            first = False

        else:
            paths = []

            for x in os.listdir(os.path.join(directory, d)):
                for y in band_mapping[sat]:
                    if x.endswith('B{}.TIF'.format(y)):
                        paths.append(os.path.join(directory, d, x))

            for tif_path in paths:
                print('warping {}'.format(os.path.basename(tif_path)))
                with rasopen(tif_path, 'r') as src:
                    with WarpedVRT(src, **vrt_options) as vrt:
                        data = vrt.read()
                        dst_dir, name = os.path.split(tif_path)
                        outfile = os.path.join(dst_dir, name)
                        meta = vrt.meta.copy()
                        meta['driver'] = 'GTiff'
                        with rasopen(outfile, 'w', **meta) as dst:
                            dst.write(data)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    images = os.path.join(home, 'landsat_images', 'LC8_39_27_test')
    warp_vrt(images, 'LC8', False)

# ========================= EOF ================================================================

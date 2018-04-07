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


import os
from datetime import datetime

from landsat import download_composer
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask
from sat_image.warped_vrt import warp_vrt


def prepare_image_stack(path, row, year, credentials, outpath, satellite='LC8'):
    start, end = '{}-05-01'.format(year), '{}-10-15'.format(year)
    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d')
    download_composer.download_landsat(start, end, satellite, path, row, outpath, credentials, max_cloud_percent=20)
    make_fmask(outpath, sat=satellite)
    warp_vrt(outpath, sat=satellite, delete_extra=True, use_band_map=True)


def make_fmask(image_dir, sat='LC8'):
    dst_path_combo = os.path.join(image_dir, 'combo_fmask.tif')
    dst_path_water = os.path.join(image_dir, 'water_fmask.tif')

    if os.path.isfile(dst_path_combo) and os.path.isfile(dst_path_water):
        print('{} and {} exist'.format(dst_path_combo, dst_path_water))

    else:
        mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}

        lst_image = mapping[sat](image_dir)

        f = Fmask(lst_image)

        cloud, shadow, water = f.cloud_mask()
        combo = f.cloud_mask(combined=True)

        f.save_array(combo, dst_path_combo)
        f.save_array(water, dst_path_water)


def orgainize_directory(pymetric_rt, paths, rows, years):
    root_list = os.listdir(pymetric_rt)
    landsat = os.path.join(pymetric_rt, 'landsat')
    if 'landsat' not in root_list:
        print('Making landsat dir in {}'.format(pymetric_rt))
        os.mkdir(landsat)
    for path in paths:
        for row in rows:
            for year in years:
                dst_dir = os.path.join(landsat, str(path), str(row),
                                       str(year))
                try:
                    os.makedirs(dst_dir)
                    print('Made {}'.format(dst_dir))
                except:
                    pass


# TODO: drop rows under cloud from PixelTrainingArray
# TODO: use pymetric function to create proper directory tree under master root (i.e., 'landsat_data')

if __name__ == '__main__':
    p, r = 39, 27
    yr = 2015
    home = os.path.expanduser('~')
    creds = os.path.join(home, 'usgs.txt')
    images = 'landsat_data'
    prepare_image_stack(p, r, yr, creds, images, satellite='LC8')

# ========================= EOF ====================================================================

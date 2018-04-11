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

from sat_image.google_download import GoogleDownload
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask
from sat_image.warped_vrt import warp_vrt


def prepare_image_stack(path, row, year, outpath, satellite=8):
    start, end = '{}-05-01'.format(year), '{}-10-15'.format(year)

    sub_directory = orgainize_directory(outpath, path, row, year)

    g = GoogleDownload(satellite, start, end, path, row, max_cloud=20)
    g.download(sub_directory)

    make_fmask(sub_directory, sat=g.sat_abv)

    warp_vrt(sub_directory, sat=g.sat_abv, delete_extra=True, use_band_map=True)

    return None


def make_fmask(image_dir, sat='LC8'):
    dst_path_cloud = os.path.join(image_dir, 'cloud_fmask.tif')
    dst_path_water = os.path.join(image_dir, 'water_fmask.tif')

    if os.path.isfile(dst_path_cloud) and os.path.isfile(dst_path_water):
        print('{} and {} exist'.format(dst_path_cloud, dst_path_water))

    else:
        mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        # TODO: configue so make_fmask gets a list of image dirs, as it stands its geting home/p/r/year
        lst_image = mapping[sat](image_dir)

        f = Fmask(lst_image)

        c, shadow, water = f.cloud_mask()
        cloud = f.cloud_mask(cloud_and_shadow=True)

        f.save_array(cloud, dst_path_cloud)
        f.save_array(water, dst_path_water)


def orgainize_directory(root, path, row, year):
    dst_dir = os.path.join(root, str(path), str(row),
                           str(year))
    if not os.path.isdir(dst_dir):
        try:
            os.makedirs(dst_dir)
            print('Made {}'.format(dst_dir))

        except:
            pass

    return dst_dir


if __name__ == '__main__':
    p, r = 39, 27
    yr = 2009
    home = os.path.expanduser('~')
    images = os.path.dirname(__file__).replace('pixel_classification', os.path.join('landsat_data'))
    prepare_image_stack(p, r, yr, images, satellite=5)

# ========================= EOF ====================================================================

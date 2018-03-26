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


import os
from datetime import datetime

from landsat import download_composer
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask


def get_image(path, row, start='2015-05-01', end='2015-10-30', usgs_credentials=None,
              output_path=None, satellite='all'):
    """ Get Landsat data from USGS site.

    :param path: int path from WRS2 reference system
    :param row: Row
    :param start: start date of image search; ftm = 'YYY-MM-DD'
    :param end: end date of image search
    :param usgs_credentials: path to .txt file with <username> <password> separated by one space
    :param output_path: location
    :param satellite: 'all' will result in search of LT5, LE7, and LC8, otherwise choose one of the three
    :return: None
    """

    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d')
    if satellite == 'all':
        for sat in ['LT5', 'LE7', 'LC8']:
            download_composer.download_landsat(start, end, sat, path=path,
                                               row=row, output_path=output_path,
                                               usgs_creds=usgs_credentials)
    else:
        download_composer.download_landsat(start, end, satellite, path=path,
                                           row=row, output_path=output_path,
                                           usgs_creds=usgs_credentials)


def make_fmask(image_dir, sat='LC8'):

    mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}

    lst_image = mapping[sat](image_dir)

    f = Fmask(lst_image)
    cloud, shadow, water = f.cloud_mask()
    combo = f.cloud_mask(combined=True)

    f.save_array(cloud, os.path.join(outdir, 'cloud_mask_l7.tif'))
    f.save_array(shadow, os.path.join(outdir, 'shadow_mask_l7.tif'))
    f.save_array(water, os.path.join(outdir, 'water_mask_l7.tif'))
    f.save_array(combo, os.path.join(outdir, 'combo_mask_l7.tif'))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    creds = os.path.join(home, 'usgs.txt')
    images = os.path.join(home, 'landsat_images', 'LC8_39_27')
    dirs = os.listdir(images)
    for image in dirs:
        make_fmask(image_dir=image)
        # get_image(39, 27, '2015-05-01', '2015-10-30',
        #           output_path=out, satellite='LC8',
        #           usgs_credentials=creds)

# ========================= EOF ====================================================================

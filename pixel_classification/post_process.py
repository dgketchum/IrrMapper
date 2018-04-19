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

from rasterio import open as rasopen
from scipy.ndimage.filters import minimum_filter

from sat_image.image import LandsatImage, Landsat8, Landsat7, Landsat5

MAPPING = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}


def filter_image(shapefile, min_filter=4):
    with rasopen(shapefile, 'r') as src:
        meta = src.meta.copy()
        data = src.read()

    filtered = minimum_filter(data, size=min_filter)
    out_name = shapefile.replace('binary_raster.tif', 'binary_filtered_4.tif')

    with rasopen(out_name, 'w', **meta) as dst:
        dst.write(filtered)

    return None


def ndvi(object):
    sat = LandsatImage(object).satellite
    lst_image = MAPPING[sat](object)
    ndvi = lst_image.ndvi()
    lst_image.save_array(ndvi, os.path.join(object, 'ndvi.tif'))
    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.dirname(__file__).replace('pixel_classification', 'landsat_data')

    year = os.path.join(root, '39', '27', '2015')
    shape = os.path.join(year, 'binary_raster.tif')
    filter_image(shape, min_filter=2)
    # ndvi(year)
# ========================= EOF ================================================================

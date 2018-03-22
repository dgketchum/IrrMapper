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
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import crs, Env


def reproject_multiband(input_file, output, dst_crs):
    with rasterio.drivers(CHECK_WITH_INVERT_PROJ=True):
        with rasterio.open(input_file) as src:
            profile = src.profile

            # Calculate the ideal dimensions and transformation in the new crs
            dst_affine, dst_width, dst_height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            # update the relevant parts of the profile
            profile.update({
                'crs': dst_crs,
                'transform': dst_affine,
                'affine': dst_affine,
                'width': dst_width,
                'height': dst_height
            })

            # Reproject and write each band
            with rasterio.open(output, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    src_array = src.read(i)
                    dst_array = np.empty((dst_height, dst_width), dtype=np.float32)

                    reproject(src_array,
                              src_crs=src.crs,
                              src_transform=src.affine,
                              destination=dst_array,
                              dst_transform=dst_affine,
                              dst_crs=dst_crs,
                              resampling=Resampling.nearest,
                              num_threads=2)

                    dst.write(dst_array, i)


if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================

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

import affine

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling


input_files = (
    # This file is in EPSG:32618
    'tests/data/RGB.byte.tif',
    # This file is in EPSG:4326
    'tests/data/WGS84-RGB.byte.tif'
)

# Destination CRS is Web Mercator
dst_crs = CRS.from_epsg(3857)

# These coordiantes are in Web Mercator
dst_bounds = -8744355, 2768114, -8559167, 2908677

# Output image dimensions
dst_height = dst_width = 100

# Output image transform
left, bottom, right, top = dst_bounds
xres = (right - left) / dst_width
yres = (top - bottom) / dst_height
dst_transform = affine.Affine(xres, 0.0, left,
                              0.0, -yres, top)

vrt_options = {
    'resampling': Resampling.cubic,
    'dst_crs': dst_crs,
    'dst_transform': dst_transform,
    'dst_height': dst_height,
    'dst_width': dst_width,
}

for path in input_files:

    with rasterio.open(path) as src:

        with WarpedVRT(src, **vrt_options) as vrt:

            # At this point 'vrt' is a full dataset with dimensions,
            # CRS, and spatial extent matching 'vrt_options'.

            # Read all data into memory.
            data = vrt.read()

            # Process the dataset in chunks.  Likely not very efficient.
            for _, window in vrt.block_windows():
                data = vrt.read(window=window)

            # Dump the aligned data into a new file.  A VRT representing
            # this transformation can also be produced by switching
            # to the VRT driver.
            directory, name = os.path.split(path)
            outfile = os.path.join(directory, 'aligned-{}'.format(name))
            rio_shutil.copy(vrt, outfile, driver='GTiff')

if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================

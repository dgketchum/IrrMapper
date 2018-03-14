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

from sat_image.image import Landsat8
from sat_image.fmask import Fmask


def fmask(directory):
    dirs = [os.path.join(directory, x) for x in os.listdir(directory)]
    tif_name = 'cloud_mask.tif'
    for d in dirs:
        if tif_name in os.listdir(d):
            print('Looks like {} already has a {}'.format(d, tif_name))
        else:
            print('Processing {}'.format(d))
            l = Landsat8(d)
            f = Fmask(l)
            combo = f.cloud_mask(combined=True)
            f.save_array(combo, outfile=os.path.join(d, tif_name))

    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    top_level = os.path.join(home, 'pixel_prep', 'irrigation',
                             'MT', 'landsat', 'LC8_39_27')
    fmask(top_level)


# ========================= EOF ====================================================================

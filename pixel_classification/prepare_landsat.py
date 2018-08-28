# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2 (the "License");
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

from landsat.google_download import GoogleDownload
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask
from sat_image.warped_vrt import warp_vrt
from sat_image.bounds import RasterBounds
from dem import AwsDem

MAPPING_OBJECTS = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
MAPPING_ABV = {5: 'LT5', 7: 'LE7', 8: 'LC8'}


def prepare_image_stack(path, row, year, outpath, satellite=8, skip_landsat=False):
    start, end = '{}-05-01'.format(year), '{}-10-15'.format(year)
    year_dir = os.path.join(outpath, str(path), str(row))
    sub_directory = orgainize_directory(outpath, path, row, year)
    print('preparing landsat at: {}'.format(sub_directory))

    if not skip_landsat:
        g = GoogleDownload(start, end, satellite, path=path, row=row,
                           output_path=sub_directory, max_cloud_percent=20)
        g.download()

    dirs = [x[0] for x in os.walk(sub_directory) if os.path.basename(x[0])[:3] in MAPPING_OBJECTS.keys()]

    master = dirs[0]
    first = True
    for d in dirs:
        satellite_abv = MAPPING_ABV[satellite]
        if first:
            l8 = MAPPING_OBJECTS[satellite_abv](master)
            dem_name = os.path.join(year_dir, 'dem.tif')
            slope_name = os.path.join(year_dir, 'slope.tif')
            if not os.path.isfile(dem_name):
                polygon = l8.get_tile_geometry()
                profile = l8.rasterio_geometry
                bb = RasterBounds(affine_transform=profile['transform'],
                                  profile=profile, latlon=True)

                dem = AwsDem(zoom=10, target_profile=profile, bounds=bb, clip_object=polygon)

                dem.terrain(attribute='elevation',
                            out_file=dem_name)
                dem.terrain(attribute='slope',
                            out_file=slope_name)
            first = False

        make_fmask(d, sat=satellite_abv)

    warp_vrt(sub_directory, delete_extra=True, use_band_map=True)

    return None


def make_fmask(image_dir, sat='LC8'):
    dst_path_cloud = os.path.join(image_dir, 'cloud_fmask.tif')
    dst_path_water = os.path.join(image_dir, 'water_fmask.tif')

    if os.path.isfile(dst_path_cloud) and os.path.isfile(dst_path_water):
        print('{} and {} exist'.format(os.path.basename(dst_path_cloud), os.path.basename(dst_path_water)))

    else:

        lst_image = MAPPING_OBJECTS[sat](image_dir)

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

        except Exception:
            pass

    return dst_dir


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================

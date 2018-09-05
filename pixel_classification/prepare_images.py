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
from ssebop_app.image import get_image


class ImageStack(object):
    """
    Prepare a stack of images from Landsat, terrain, etc. Save stack in identical geometry.
    """

    def __init__(self, satellite, path, row, root=None, max_cloud_pct=None, start=None, end=None,
                 year=None, output_directory=None):

        self.landsat_mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        self.landsat_mapping_abv = {5: 'LT5', 7: 'LE7', 8: 'LC8'}

        self.sat = satellite
        self.sat_abv = self.landsat_mapping_abv[self.sat]
        self.sat_obj = self.landsat_mapping[self.sat_abv]

        self.path = path
        self.row = row
        self.year = year

        self.max_cloud = max_cloud_pct
        self.start = start
        self.end = end
        self.root = root
        self.out_dir = output_directory

        self.profile = None
        self.image_dir = None
        self.dst_path_cloud = None
        self.dst_path_water = None
        self.landsat = None
        self.image_dirs = None
        self.image_paths = None

        if start and end:
            self.start_dt = '{}-05-01'.format(self.year)
            self.end_dt = '{}-10-15'.format(self.year)

    def get_landsat(self):
        g = GoogleDownload(self.start, self.end, self.sat, path=self.path, row=self.row,
                           output_path=self.out_dir, max_cloud_percent=self.max_cloud)
        g.download()

        self.image_dirs = [x[0] for x in os.walk(self.out_dir) if
                           os.path.basename(x[0])[:3] in self.landsat_mapping.keys()]

        self.get_geography()
        [self.make_fmask(d) for d in self.image_dirs]

    def get_geography(self):

        master = self.image_dirs[0]
        self.landsat = self.landsat_mapping[self.sat_abv](master)

    def get_terrain(self):

        slope_name = os.path.join(self.out_dir, 'slope.tif')

        if not os.path.isfile(slope_name):
            polygon = self.landsat.get_tile_geometry()
            self.profile = self.landsat.rasterio_geometry

            bb = RasterBounds(affine_transform=self.profile['transform'],
                              profile=self.profile, latlon=True)

            dem = AwsDem(zoom=10, target_profile=self.profile, bounds=bb,
                         clip_object=polygon)

            dem.terrain(attribute='slope',
                        out_file=slope_name)

    def get_et(self):
        get_image(self.out_dir, self.root, image_exists=True, satellite=self.sat, path=self.path,
                  row=self.row)

    def warp_vrt(self):
        warp_vrt(self.out_dir, delete_extra=True, use_band_map=True)

    def make_fmask(self, image_dir):
        self.dst_path_cloud = os.path.join(image_dir, 'cloud_fmask.tif')
        self.dst_path_water = os.path.join(image_dir, 'water_fmask.tif')

        if os.path.isfile(self.dst_path_cloud) and os.path.isfile(self.dst_path_water):
            print('{} and {} exist'.format(os.path.basename(self.dst_path_cloud),
                                           os.path.basename(self.dst_path_water)))

        else:

            lst_image = self.landsat_mapping[sat](image_dir)

            f = Fmask(lst_image)

            c, shadow, water = f.cloud_mask()
            cloud = f.cloud_mask(cloud_and_shadow=True)

            f.save_array(cloud, self.dst_path_cloud)
            f.save_array(water, self.dst_path_water)

    def orgainize_directory(self):
        dst_dir = os.path.join(self.root, str(self.path), str(self.row),
                               str(self.year))
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

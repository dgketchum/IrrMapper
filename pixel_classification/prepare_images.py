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
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
from numpy import mean

from landsat.google_download import GoogleDownload
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask
from sat_image.warped_vrt import warp_vrt
from bounds import RasterBounds
from dem import AwsDem
from ssebop_app.image import get_image

from pixel_classification.crop_data_layer import CropDataLayer as Cdl


class ImageStack(object):
    """
    Prepare a stack of images from Landsat, terrain, etc. Save stack in identical geometry.
    """

    def __init__(self, satellite, path, row, root=None, max_cloud_pct=None, start=None, end=None,
            year=None, n_landsat=None):

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

        self.profile = None
        self.dst_path_cloud = None
        self.dst_path_water = None
        self.landsat = None
        self.scenes = None
        self.image_dirs = None
        self.image_paths = None

        self.cdl_tif = None
        self.cdl_mask = None

        self.n = n_landsat

        self.ancillary_rasters = []

        if year and not start and not end:
            self.start = '{}-05-01'.format(self.year)
            self.end = '{}-10-15'.format(self.year)

    def build_all(self):
        self.get_landsat()
        self.profile = self.landsat.rasterio_geometry
        self.get_et()
        self.get_terrain()
        self.get_cdl()

    def get_cdl(self):
        self.cdl_mask = os.path.join(self.root, 'cdl_mask.tif')
        if not os.path.isfile(self.cdl_mask):
            print('get {}'.format(self.cdl_mask))
            polygon = self.landsat.get_tile_geometry()
            cdl = Cdl(year=self.year, target_profile=self.landsat.profile)
            cdl.get_mask(clip_geometry=polygon, out_file=self.cdl_mask)
        else:
            print('{} exists'.format(self.cdl_mask))

    def get_landsat(self):
        g = GoogleDownload(self.start, self.end, self.sat, path=self.path, row=self.row,
                           output_path=self.root, max_cloud_percent=self.max_cloud)

        g.select_scenes(self.n)
        self.scenes = g.selected_scenes
        g.download(list_type='selected')

        self.image_dirs = [x[0] for x in os.walk(self.root) if
                           os.path.basename(x[0])[:3] in self.landsat_mapping.keys()]

        self._get_geography()
        [self._make_fmask(d) for d in self.image_dirs]

    def get_terrain(self):

        slope_name = os.path.join(self.root, 'slope.tif')
        aspect_name = os.path.join(self.root, 'aspect.tif')
        dif_elev = os.path.join(self.root, 'elevation_diff.tif')

        self.ancillary_rasters.append(slope_name)
        self.ancillary_rasters.append(aspect_name)
        self.ancillary_rasters.append(dif_elev)

        check = [os.path.isfile(x) for x in [slope_name, aspect_name, dif_elev]]

        if False in check:
            polygon = self.landsat.get_tile_geometry()
            bb = RasterBounds(affine_transform=self.profile['transform'],
                              profile=self.profile, latlon=True)
            dem = AwsDem(zoom=10, target_profile=self.profile, bounds=bb,
                         clip_object=polygon)

            dem.terrain(attribute='slope',
                        out_file=slope_name, save_and_return=True)
            dem.terrain(attribute='aspect',
                        out_file=aspect_name, save_and_return=True)
            elev = dem.terrain(attribute='elevation')
            elev = elev - mean(elev)
            dem.save(elev, geometry=dem.target_profile, output_filename=dif_elev)

    def get_et(self):
        for i, d in enumerate(self.image_dirs):
            l = self.landsat_mapping[self.sat_abv](d)
            _id = l.landsat_scene_id
            get_image(image_dir=d, parent_dir=self.root, image_exists=True, image_id=_id,
                      satellite=self.sat, path=self.path, row=self.row, image_date=l.date_acquired,
                      landsat_object=self.landsat, overwrite=False)
            products = ['lst', 'ssebop_etrf']
            for p in products:
                self.ancillary_rasters.append(os.path.join(d, '{}_{}.tif'.format(_id, p)))

    def warp_vrt(self):
        warp_vrt(self.root, delete_extra=False, use_band_map=False, remove_bqa=True)

    def _get_geography(self):

        master = self.image_dirs[0]
        self.landsat = self.landsat_mapping[self.sat_abv](master)

    def _make_fmask(self, image_dir):

        self.dst_path_cloud = os.path.join(image_dir, 'cloud_fmask.tif')
        self.dst_path_water = os.path.join(image_dir, 'water_fmask.tif')

        if os.path.isfile(self.dst_path_cloud) and os.path.isfile(self.dst_path_water):
            print('{} and {} exist for {}'.format(os.path.basename(self.dst_path_cloud),
                                                  os.path.basename(self.dst_path_water),
                                                  image_dir))

        else:
            print('fmask for {}'.format(image_dir))
            lst_image = self.landsat_mapping[self.sat_abv](image_dir)

            f = Fmask(lst_image)

            c, shadow, water = f.cloud_mask()
            cloud = c | shadow

            f.save_array(cloud, self.dst_path_cloud)
            f.save_array(water, self.dst_path_water)

    def _orgainize_directory(self):
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

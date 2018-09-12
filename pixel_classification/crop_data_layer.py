# ===============================================================================
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
# ===============================================================================

import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)

from requests import get
import copy
from numpy import empty, float32, array
from rasterio import open as rasopen
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform as cdt
from tempfile import mkdtemp
from bounds import RasterBounds


class CropDataLayer(object):

    def __init__(self, target_profile, year):
        self.url_base = 'ftp.nass.usda.gov/download/res'
        self.url_year = '{}_30m_cdls.zip'.format(year)
        self.url = os.path.join(self.url_base, self.url_year)
        self.cdl_location = os.path.join(os.path.dirname(__file__), 'model_data')
        self.zip_file = os.path.join(os.path.dirname(__file__), 'model_data', '{}_30m_cdls.zip'.format(year))

        self.temp_dir = mkdtemp()

        self.target_profile = target_profile
        self.bbox = RasterBounds(profile=self.target_profile)
        self.src_bounds_wsen = None

        self.projection = None
        self.reprojection = None

    def download_zipped_cdl(self):
        if not os.path.isfile(self.zip_file):
            req = get(self.url, stream=True)
            if req.status_code != 200:
                raise ValueError('Bad response {} from request.'.format(req.status_code))

            with open(self.zip_file, 'wb') as f:
                print('Downloading {}'.format(self.url))
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

    def conform(self, subset, out_file=None):
        if subset.dtype != float32:
            subset = array(subset, dtype=float32)
        self._project(subset)
        self._reproject()
        self._mask()
        result = self._resample()
        if out_file:
            self.save_raster(result, self.target_profile, output_filename=out_file)
        return result

    def _project(self, subset):

        proj_path = os.path.join(self.temp_dir, 'tiled_proj.tif')
        setattr(self, 'projection', proj_path)

        profile = copy.deepcopy(self.target_profile)
        profile['dtype'] = float32
        bb = self.bbox.as_tuple()

        if self.src_bounds_wsen:
            bounds = self.src_bounds_wsen
        else:
            bounds = (bb[0], bb[1],
                      bb[2], bb[3])

        dst_affine, dst_width, dst_height = cdt(CRS({'init': 'epsg:4326'}),
                                                CRS({'init': 'epsg:4326'}),
                                                subset.shape[1],
                                                subset.shape[2],
                                                *bounds)

        profile.update({'crs': CRS({'init': 'epsg:4326'}),
                        'transform': dst_affine,
                        'width': dst_width,
                        'height': dst_height})

        with rasopen(proj_path, 'w', **profile) as dst:
            dst.write(subset)

    def _reproject(self):

        reproj_path = os.path.join(self.temp_dir, 'reproj.tif')
        setattr(self, 'reprojection', reproj_path)

        with rasopen(self.projection, 'r') as src:
            src_profile = src.profile
            src_bounds = src.bounds
            src_array = src.read(1)

        dst_profile = copy.deepcopy(self.target_profile)
        dst_profile['dtype'] = float32
        bounds = src_bounds
        dst_affine, dst_width, dst_height = cdt(src_profile['crs'],
                                                dst_profile['crs'],
                                                src_profile['width'],
                                                src_profile['height'],
                                                *bounds)

        dst_profile.update({'crs': dst_profile['crs'],
                            'transform': dst_affine,
                            'width': dst_width,
                            'height': dst_height})

        with rasopen(reproj_path, 'w', **dst_profile) as dst:
            dst_array = empty((1, dst_height, dst_width), dtype=float32)

            reproject(src_array, dst_array, src_transform=src_profile['transform'],
                      src_crs=src_profile['crs'], dst_crs=self.target_profile['crs'],
                      dst_transform=dst_affine, resampling=Resampling.nearest,
                      num_threads=2)

            dst.write(dst_array.reshape(1, dst_array.shape[1], dst_array.shape[2]))

    def _mask(self):

        mask_path = os.path.join(self.temp_dir, 'masked.tif')

        with rasopen(self.reprojection) as src:
            out_arr, out_trans = mask(src, self.clip_feature, crop=True,
                                      all_touched=True)
            out_meta = src.meta.copy()
            out_meta.update({'driver': 'GTiff',
                             'height': out_arr.shape[1],
                             'width': out_arr.shape[2],
                             'transform': out_trans})

        with rasopen(mask_path, 'w', **out_meta) as dst:
            dst.write(out_arr)

        setattr(self, 'mask', mask_path)
        delattr(self, 'reprojection')

    def _resample(self):

        # home = os.path.expanduser('~')
        # resample_path = os.path.join(home, 'images', 'sandbox', 'thredds', 'resamp_twx_{}.tif'.format(var))

        resample_path = os.path.join(self.temp_dir, 'resample.tif')

        with rasopen(self.mask, 'r') as src:
            array = src.read(1)
            profile = src.profile
            res = src.res
            try:
                target_affine = self.target_profile['affine']
            except KeyError:
                target_affine = self.target_profile['transform']
            target_res = target_affine.a
            res_coeff = res[0] / target_res

            new_array = empty(shape=(1, round(array.shape[0] * res_coeff),
                                     round(array.shape[1] * res_coeff)), dtype=float32)
            aff = src.affine
            new_affine = Affine(aff.a / res_coeff, aff.b, aff.c, aff.d, aff.e / res_coeff, aff.f)

            profile['transform'] = self.target_profile['transform']
            profile['width'] = self.target_profile['width']
            profile['height'] = self.target_profile['height']
            profile['dtype'] = str(new_array.dtype)

            delattr(self, 'mask')

            with rasopen(resample_path, 'w', **profile) as dst:
                reproject(array, new_array, src_transform=aff, dst_transform=new_affine, src_crs=src.crs,
                          dst_crs=src.crs, resampling=Resampling.nearest)

                dst.write(new_array)

            with rasopen(resample_path, 'r') as src:
                arr = src.read()

            return arr

    @staticmethod
    def save_raster(arr, geometry, output_filename):
        try:
            arr = arr.reshape(1, arr.shape[1], arr.shape[2])
        except IndexError:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        geometry['dtype'] = str(arr.dtype)

        with rasopen(output_filename, 'w', **geometry) as dst:
            dst.write(arr)
        return None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

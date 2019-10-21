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

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import re
import copy
from xml.etree import ElementTree
from requests import get
from numpy import empty, float32
from numpy import isin
from rasterio import open as rasopen
from rasterio.dtypes import uint8
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform as cdt
from tempfile import mkdtemp
from bounds import RasterBounds


class CropDataLayer(object):

    def __init__(self, target_profile=None, year=None, out_dir=None, from_file=None):

        self.url_base = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/' \
                        'GetCDLFile?year={year}&bbox={wsen}'

        if from_file:
            self.from_file = from_file
            with rasopen(from_file) as src:
                self.cdl = src.read()
                self.target_profile = src.profile
                self.cdl_empty = False

        else:
            self.cdl_empty = True
            self.cdl = None
            if not out_dir:
                self.cdl_location = os.path.join(os.path.dirname(__file__), 'model_data')
            else:
                self.cdl_location = out_dir

            self.zip_file = os.path.join(self.cdl_location, '{}_30m_cdls.zip'.format(year))

            self.temp_dir = mkdtemp()

            if target_profile and year:
                self.target_profile = target_profile
                self.bbox = RasterBounds(profile=self.target_profile,
                                         affine_transform=self.target_profile['transform'])
                self.bbox.expand(**{'east': 0.1, 'west': -0.1, 'north': 0.2, 'south': -0.2})
                self.bbox_projected = bb = self.bbox.to_lambert_conformal_conic()
                bb_str = '{},{},{},{}'.format(bb[0], bb[1], bb[2], bb[3])
                self.request_url = self.url_base.format(year=year, wsen=bb_str)
                self.data_url = self._get_data_url()

            self.original_tif = None
            self.mask = None
            self.projection = None
            self.reprojection = None

    def get_original_tif(self, out_file=None):

        req = get(self.data_url, verify=False)

        if req.status_code != 200:
            raise ValueError('Bad response {} from request.'.format(req.status_code))

        if not out_file:
            self.original_tif = os.path.join(self.temp_dir, os.path.basename(self.data_url))
        else:
            self.original_tif = out_file

        with open(self.original_tif, 'wb') as f:
            print('Downloading {}'.format(self.data_url))
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    def get_conforming_data(self, clip_geometry, keep_original=False, out_file=None):
        self.get_original_tif()
        self._reproject()
        self._mask(clip_geometry)
        result = self._resample()

        if not keep_original:
            os.remove(self.original_tif)

        if out_file:
            self.save(result, self.target_profile, output_filename=os.path.join(self.cdl_location,
                                                                                'cdl.tif'))
        self.cdl = result
        return result

    def get_mask(self, clip_geometry=None, out_file=None):

        arr = None

        if self.cdl_empty:
            try:
                arr = self.get_conforming_data(clip_geometry=clip_geometry)
            except ValueError:
                print('Need clip geometry to build cdl')
        else:
            arr = self.cdl

        crop = list(self.crop.keys())
        msk = isin(arr, crop)
        msk = ~msk
        msk = msk.astype(uint8)
        profile = copy.deepcopy(self.target_profile)
        profile['dtype'] = uint8
        if out_file:
            with rasopen(out_file, 'w', **profile) as dst:
                dst.write(msk)

        return msk

    def _reproject(self):

        self.reprojection = os.path.join(self.temp_dir, 'cdl_reprojection.tif')

        with rasopen(self.original_tif, 'r') as src:
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

        with rasopen(self.reprojection, 'w', **dst_profile) as dst:
            dst_array = empty((1, dst_height, dst_width), dtype=float32)

            reproject(src_array, dst_array, src_transform=src_profile['transform'],
                      src_crs=src_profile['crs'], dst_crs=self.target_profile['crs'],
                      dst_transform=dst_affine, resampling=Resampling.nearest,
                      num_threads=2)

            dst.write(dst_array.reshape(1, dst_array.shape[1], dst_array.shape[2]))

    def _mask(self, clip):

        mask_path = os.path.join(self.temp_dir, 'masked.tif')

        with rasopen(self.reprojection) as src:
            out_arr, out_trans = mask(src, clip, crop=True,
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
            aff = src.transform
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
    def save(array, geometry, output_filename, crs=None, return_array=False):
        try:
            array = array.reshape(1, array.shape[1], array.shape[2])
        except IndexError:
            array = array.reshape(1, array.shape[0], array.shape[1])
        geometry['dtype'] = str(array.dtype)
        if crs:
            geometry['crs'] = CRS({'init': crs})
        with rasopen(output_filename, 'w', **geometry) as dst:
            dst.write(array)
        if return_array:
            return array
        return None

    def download_zipped_cdl(self):
        if not os.path.isfile(self.zip_file):
            req = urlretrieve(self.request_url, self.cdl_location)
            if req.status_code != 200:
                raise ValueError('Bad response {} from request.'.format(req.status_code))

            with open(self.zip_file, 'wb') as f:
                print('Downloading {}'.format(self.request_url))
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

    def _get_data_url(self):
        r = get(self.request_url, verify=False)
        tree = ElementTree.fromstring(r.content)
        u = [ElementTree.tostring(e) for e in tree][0].decode("utf-8")
        result = re.search('<returnURL>(.*)</returnURL>', u).group(1)
        return result

    @property
    def crop(self):
        return {1: 'Corn',
                2: 'Cotton',
                3: 'Rice',
                4: 'Sorghum',
                5: 'Soybeans',
                6: 'Sunflower',
                10: 'Peanuts',
                11: 'Tobacco',
                12: 'Sweet Corn',
                13: 'Pop or Orn Corn',
                14: 'Mint',
                21: 'Barley',
                22: 'Durum Wheat',
                23: 'Spring Wheat',
                24: 'Winter Wheat',
                25: 'Other Small Grains',
                26: 'Dbl Crop WinWht / Soybeans',
                27: 'Rye',
                28: 'Oats',
                29: 'Millet',
                30: 'Speltz',
                31: 'Canola',
                32: 'Flaxseed',
                33: 'Safflower',
                34: 'Rape Seed',
                35: 'Mustard',
                36: 'Alfalfa',
                37: 'Other Hay / NonAlfalfa',
                38: 'Camelina',
                39: 'Buckwheat',
                41: 'Sugarbeets',
                42: 'Dry Beans',
                43: 'Potatoes',
                44: 'Other Crops',
                45: 'Sugarcane',
                46: 'Sweet Potatoes',
                47: 'Misc Vegs & Fruits',
                48: 'Watermelons',
                49: 'Onions',
                50: 'Cucumbers',
                51: 'Chick Peas',
                52: 'Lentils',
                53: 'Peas',
                54: 'Tomatoes',
                55: 'Caneberries',
                56: 'Hops',
                57: 'Herbs',
                58: 'Clover/Wildflowers',
                61: 'Fallow/Idle Cropland',
                66: 'Cherries',
                67: 'Peaches',
                68: 'Apples',
                69: 'Grapes',
                70: 'Christmas Trees',
                71: 'Other Tree Crops',
                72: 'Citrus',
                74: 'Pecans',
                75: 'Almonds',
                76: 'Walnuts',
                77: 'Pears',
                204: 'Pistachios',
                205: 'Triticale',
                206: 'Carrots',
                207: 'Asparagus',
                208: 'Garlic',
                209: 'Cantaloupes',
                210: 'Prunes',
                211: 'Olives',
                212: 'Oranges',
                213: 'Honeydew Melons',
                214: 'Broccoli',
                216: 'Peppers',
                217: 'Pomegranates',
                218: 'Nectarines',
                219: 'Greens',
                220: 'Plums',
                221: 'Strawberries',
                222: 'Squash',
                223: 'Apricots',
                224: 'Vetch',
                225: 'Dbl Crop WinWht/Corn',
                226: 'Dbl Crop Oats/Corn',
                227: 'Lettuce',
                229: 'Pumpkins',
                230: 'Dbl Crop Lettuce/Durum Wht',
                231: 'Dbl Crop Lettuce/Cantaloupe',
                232: 'Dbl Crop Lettuce/Cotton',
                233: 'Dbl Crop Lettuce/Barley',
                234: 'Dbl Crop Durum Wht/Sorghum',
                235: 'Dbl Crop Barley/Sorghum',
                236: 'Dbl Crop WinWht/Sorghum',
                237: 'Dbl Crop Barley/Corn',
                238: 'Dbl Crop WinWht/Cotton',
                239: 'Dbl Crop Soybeans/Cotton',
                240: 'Dbl Crop Soybeans/Oats',
                241: 'Dbl Crop Corn/Soybeans',
                242: 'Blueberries',
                243: 'Cabbage',
                244: 'Cauliflower',
                245: 'Celery',
                246: 'Radishes',
                247: 'Turnips',
                248: 'Eggplants',
                249: 'Gourds',
                250: 'Cranberries',
                254: 'Dbl Crop Barley/Soybeans'}

    @property
    def non_crop(self):
        return {37: 'Other Hay/Non Alfalfa',
                59: 'Sod/Grass Seed',
                60: 'Switchgrass',
                63: 'Forest',
                64: 'Shrubland',
                65: 'Barren',
                81: 'Clouds/No Data',
                82: 'Developed',
                83: 'Water',
                87: 'Wetlands',
                88: 'Nonag/Undefined',
                92: 'Aquaculture',
                111: 'Open Water',
                112: 'Perennial Ice/Snow',
                121: 'Developed/Open Space',
                122: 'Developed/Low Intensity',
                123: 'Developed/Med Intensity',
                124: 'Developed/High Intensity',
                131: 'Barren',
                141: 'Deciduous Forest',
                142: 'Evergreen Forest',
                143: 'Mixed Forest',
                152: 'Shrubland',
                176: 'Grass/Pasture',
                190: 'Woody Wetlands',
                195: 'Herbaceous Wetlands'}


if __name__ == '__main__':
    print(CropDataLayer.cdl.keys())
    pass
# ========================= EOF ====================================================================

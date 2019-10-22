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
from datetime import datetime

from bounds import GeoBounds, BufferPoint
from numpy import empty, uint8
from rasterio import Env
from rasterio import open as rasopen
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform
from requests import get


class BadCoordinatesError(ValueError):
    pass


class MissingArgumentError(ValueError):
    pass


class NaipImage(object):
    def __init__(self):
        self.profile = None
        self.array = None
        self.web_mercator_bounds = None

        temp_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

        self.temp_file = os.path.join(temp_dir, 'tile.tif')
        self.temp_proj_file = os.path.join(os.getcwd(), 'temp', 'tile_projected.tif')

    def save(self, array, geometry, output_filename, crs=None):
        array = array.reshape(geometry['count'], array.shape[1], array.shape[2])
        geometry['dtype'] = uint8

        if crs:
            dst_crs = CRS({'init': 'epsg:{}'.format(crs)})
            if geometry['crs'] != dst_crs:
                self.reproject_multiband(output_filename, dst_crs)
                return None

        with rasopen(output_filename, 'w', **geometry) as dst:
            dst.write(array)

        return None

    def reproject_multiband(self, file_name, dst_crs):
        with Env(CHECK_WITH_INVERT_PROJ=True):
            with rasopen(self.temp_file) as src:
                profile = src.profile

                dst_affine, dst_width, dst_height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)

                profile.update({
                    'crs': dst_crs,
                    'transform': dst_affine,
                    'width': dst_width,
                    'height': dst_height
                })

                with rasopen(file_name, 'w', **profile) as dst:
                    for i in range(1, src.count + 1):
                        src_array = src.read(i)
                        dst_array = empty((dst_height, dst_width), dtype=uint8)

                        reproject(src_array,
                                  src_crs=src.crs,
                                  src_transform=src.transform,
                                  destination=dst_array,
                                  dst_transform=dst_affine,
                                  dst_crs=dst_crs,
                                  resampling=Resampling.nearest,
                                  num_threads=2)

                        dst.write(dst_array, i)


class ApfoNaip(NaipImage):
    """  APFO web service NAIP image object.

    See query options, pass kwargs dict with following keys:

    '&bboxSR='
    '&size='
    '&imageSR='
    '&time='
    '&format=tiff'
    '&pixelType=U8'
    '&noData='
    '&noDataInterpretation=esriNoDataMatchAny'
    '&interpolation=+RSP_BilinearInterpolation'
    '&compression='
    '&compressionQuality='
    '&bandIds='
    '&mosaicRule='
    '&renderingRule='
    '&f=html'

    """

    def __init__(self, bbox=None, **kwargs):
        """
            :param bbox: (west, south, east, north) tuple in geographic coordinates
        """

        # TODO un-hard-code pixelType, bboxSR, etc

        NaipImage.__init__(self)

        self.bbox = bbox
        self.dst_crs = None

        if not bbox and not kwargs['centroid']:
            raise NotImplementedError('Must provide either a bounding box (w, s, e, n) or centroid in Geographic')
        if 'centroid' in kwargs.keys():
            lat = kwargs['centroid'][0]
            lon = kwargs['centroid'][1]
            if 'buffer' not in kwargs.keys():
                kwargs['buffer'] = 1000
            self.bbox = bbox = BufferPoint().buffer_meters(lat, lon, kwargs['buffer'])

        if abs(bbox[0]) > 180 or abs(bbox[1]) > 90:
            raise BadCoordinatesError('{} is not a good latitude'.format(bbox[0]))

        if 'year' in kwargs.keys():
            y = int(kwargs['year'])
            yr_ep = str(int((datetime(y, 1, 1, 0, 0, 0) - datetime(1970, 1, 1)).total_seconds()))
            epoch = '{}%2C1262304000000'.format(yr_ep).

        else:
            epoch = '1230768000000%2C1262304000000'

        self.naip_base_url = 'https://gis.apfo.usda.gov/arcgis/rest/services/NAIP_Historical/'
        self.usda_query_str = '{a}/ImageServer/exportImage?f=image&time={b}&bbox={a}' \
                              '&imageSR=102100&bboxSR=102100&size=1024,1024' \
                              '&format=tiff&pixelType=U8' \
                              '&interpolation=+RSP_BilinearInterpolation'.format(a='{}', b=epoch)

        for key, val in kwargs.items():
            self.__setattr__(key, val)

        self.bounds_fmt = '{w},{s},{e},{n}'

    def get_image(self, state):
        """ Get NAIP imagery from states excluding Hawaii and Alaska

        Current hack in this method and in GeoBounds is hard-coded epsg: 3857 'web mercator',
        though the NAIP service provides epsg: 102100 a deprecated ESRI SRS'

        :param state: e.g. 'ND'
        :param size: tuple of horizontal by vertical size in pixels, e.g., (512, 512)
        :return:
        """

        coords = {x: y for x, y in zip(['west', 'south', 'east', 'north'], self.bbox)}

        w, s, e, n = GeoBounds(**coords).to_web_mercator()
        self.web_mercator_bounds = (w, s, e, n)

        bbox_str = self.bounds_fmt.format(w=w, s=s, e=e, n=n)

        naip_str = '{}_NAIP'.format(state)
        query = self.usda_query_str.format(naip_str, bbox_str)
        url = '{}{}'.format(self.naip_base_url, query)

        req = get(url, verify=False, stream=True)
        if req.status_code != 200:
            raise ValueError('Bad response {} from NAIP API request.'.format(req.status_code))
        with open(self.temp_file, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        with rasopen(self.temp_file, 'r') as src:
            array = src.read()
            profile = src.profile

        return array, profile

    def close(self):
        if os.path.isfile(self.temp_file):
            os.remove(self.temp_file)
        if os.path.isfile(self.temp_proj_file):
            os.remove(self.temp_proj_file)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    pass

# ========================= EOF ================================================================

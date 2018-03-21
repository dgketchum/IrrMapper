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
from requests import get
from rasterio import open as rasopen
from rasterio.crs import CRS

from spatial.naip_services import get_naip_key
from spatial.bounds import GeoBounds


class NaipImage(object):
    def __init__(self):
        pass

    @staticmethod
    def save(array, geometry, output_filename, crs=None):
        try:
            array = array.reshape(1, array.shape[1], array.shape[2])
        except IndexError:
            array = array.reshape(1, array.shape[0], array.shape[1])
        geometry['dtype'] = array.dtype
        if crs:
            geometry['crs'] = CRS({'init': crs})
        with rasopen(output_filename, 'w', **geometry) as dst:
            dst.write(array)
        return None


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
        NaipImage.__init__(self)
        self.bbox = bbox
        self.naip_base_url = 'https://gis.apfo.usda.gov/arcgis/rest/services/'
        self.usda_query_str = '{a}/ImageServer/exportImage?f=image&bbox={a}&imageSR={a}' \
                              '&bboxSR={a}&format=tiff&pixelType=U8&size={a},{a}' \
                              '&interpolation=+RSP_BilinearInterpolation'.format(a='{}')
        self.query_kwargs = ''
        for key, val in kwargs.items():
            self.__setattr__(key, val)
            self.query_kwargs += '&{}={}'.format(key, val)

        self.bounds_fmt = '{w},{s},{e},{n}'

    def get_image(self, state, size=(512, 512)):
        """ Get NAIP imagery from states excluding Hawaii and Alaska

        Current hack in this method and in GeoBounds is hard-coded epsg: 3857 'web mercator',
        though the NAIP service provides epsg: 102100 a deprecated ESRI SRS'

        :param state: lower case state str, e.g. 'south_dakota'
        :param size: tuple of horizontal by vertical size in pixels, e.g., (526, 525)
        :return:
        """
        coords = {x: y for x, y in zip(['west', 'south', 'east', 'north'], self.bbox)}
        srs = 102100

        w, s, e, n = GeoBounds(**coords).to_web_mercator()
        bbox_str = self.bounds_fmt.format(w=w, s=s, e=e, n=n)
        nh, nv = size

        naip_str = get_naip_key(state)
        query = self.usda_query_str.format(naip_str, bbox_str, srs, srs, nh, nv)
        url = '{}{}'.format(self.naip_base_url, query)
        req = get(url, verify=False, stream=True)
        if req.status_code != 200:
            raise ValueError('Bad response from NAIP API request.')
        temp = os.path.join(os.getcwd(), 'temp', 'tile.tif')
        with open(temp, 'wb') as f:
            f.write(req.content)
        with rasopen(temp, 'r') as src:
            array = src.read()
            profile = src.profile
        os.remove(temp)
        pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tile_size = (512, 512)
    box = (-110.08, 46.256, -109.61, 46.584)
    naip = ApfoNaip(box)
    naip.get_image('montana', tile_size)

# ========================= EOF ================================================================

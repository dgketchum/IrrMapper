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
from tempfile import mkdtemp

from spatial.naip_services import get_naip_key
from spatial.bounds import GeoBounds

NAIP_BASE = 'https://gis.apfo.usda.gov/arcgis/rest/services/'
USDA_NAIP = '{a}/ImageServer/exportImage?f=image&bbox={a}&imageSR={a}' \
            '&bboxSR={a}&size={a},{a}'.format(a='{}')
BBOX_fmt = '{w},{s},{e},{n}'

arc_url = 'https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/Montana_2015_1m/ImageServer/' \
          'exportImage?f=image' \
          '&bbox=-12255105.148460371%2C5843797.028192342%2C-12249706.783337737%2C5849682.679370292' \
          '&imageSR=102100&bboxSR=102100&size=565%2C616'


def get_naip_image(state, bbox, size=(512, 512)):
    """ Get NAIP imagery from states excluding Hawaii and Alaska

    :param state: lower case state str, e.g. 'south_dakota'
    :param bbox: (west, south, east, north) tuple in geographic coordinates
    :param size: tuple of horizontal by vertical size in pixels, e.g., (526, 525)
    :param spatial_ref:
    :param input: input coordinate reference system epsg code
    :return:
    """
    coords = {x: y for x, y in zip(['west', 'south', 'east', 'north'], bbox)}
    srs = 102100
    w, s, e, n = GeoBounds(**coords).to_web_mercator()
    nh, nv = size
    naip_str = get_naip_key(state)
    bbox_str = BBOX_fmt.format(w=w, s=s, e=e, n=n)
    query = USDA_NAIP.format(naip_str, bbox_str, srs, srs, nh, nv)
    url = '{}{}'.format(NAIP_BASE, query)
    req = get(url, verify=False, stream=True)
    if req.status_code != 200:
        raise ValueError('Bad response from Mapzen API request.')
    temp = os.path.join(os.getcwd(), 'tile.tif')
    with open(temp, 'wb') as f:
        f.write(req.content)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    tile_size = (512, 512)
    box = (-110.08, 46.256, -109.61, 46.584)
    get_naip_image('montana', box, tile_size)

# ========================= EOF ================================================================

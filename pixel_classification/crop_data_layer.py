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


class CropDataLayer(object):

    def __init__(self, year):
        self.url_base = 'ftp.nass.usda.gov/download/res'
        self.url_year = '{}_30m_cdls.zip'.format(year)
        self.url = os.path.join(self.url_base, self.url_year)
        self.cdl_location = os.path.join(os.path.dirname(__file__), 'model_data')
        self.zip_file = os.path.join(os.path.dirname(__file__), 'model_data', '{}_30m_cdls.zip'.format(year))

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

        return None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF (the "License");
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
import json

NAIP_JSON = {"currentVersion": 10.2, "folders": [], "services": [{"name": "NAIP/Alabama", "type": "ImageServer"},
                                                                 {"name": "NAIP/Arizona", "type": "ImageServer"},
                                                                 {"name": "NAIP/Arkansas", "type": "ImageServer"},
                                                                 {"name": "NAIP/California_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Colorado", "type": "ImageServer"},
                                                                 {"name": "NAIP/Connecticut_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Delaware", "type": "ImageServer"},
                                                                 {"name": "NAIP/Florida", "type": "ImageServer"},
                                                                 {"name": "NAIP/Georgia", "type": "ImageServer"},
                                                                 {"name": "NAIP/Idaho", "type": "ImageServer"},
                                                                 {"name": "NAIP/Illinois", "type": "ImageServer"},
                                                                 {"name": "NAIP/Indiana_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Iowa", "type": "ImageServer"},
                                                                 {"name": "NAIP/Kansas", "type": "ImageServer"},
                                                                 {"name": "NAIP/Kentucky_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Louisiana", "type": "ImageServer"},
                                                                 {"name": "NAIP/Maine_2015_1m", "type": "ImageServer"},
                                                                 {"name": "NAIP/Maryland", "type": "ImageServer"},
                                                                 {"name": "NAIP/Massachusetts_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Michigan_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Minnesota", "type": "ImageServer"},
                                                                 {"name": "NAIP/Mississippi_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Missouri_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Montana_2015_1m",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Nebraska_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Nevada", "type": "ImageServer"},
                                                                 {"name": "NAIP/New_Hampshire_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/New_Jersey", "type": "ImageServer"},
                                                                 {"name": "NAIP/New_Mexico_2016_1m",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/New_York", "type": "ImageServer"},
                                                                 {"name": "NAIP/North_Carolina_2016_1m",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/North_Dakota", "type": "ImageServer"},
                                                                 {"name": "NAIP/Ohio", "type": "ImageServer"},
                                                                 {"name": "NAIP/Oklahoma", "type": "ImageServer"},
                                                                 {"name": "NAIP/Oregon_2016_1m", "type": "ImageServer"},
                                                                 {"name": "NAIP/Pennsylvania", "type": "ImageServer"},
                                                                 {"name": "NAIP/Rhode_Island_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/South_Carolina", "type": "ImageServer"},
                                                                 {"name": "NAIP/South_Dakota_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Tennessee_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Texas_2016_1m", "type": "ImageServer"},
                                                                 {"name": "NAIP/Utah_2016_1m", "type": "ImageServer"},
                                                                 {"name": "NAIP/Vermont_2016_60cm",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Virginia_2016_1m",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Washington", "type": "ImageServer"},
                                                                 {"name": "NAIP/West_Virginia_2016_1m",
                                                                  "type": "ImageServer"},
                                                                 {"name": "NAIP/Wisconsin", "type": "ImageServer"},
                                                                 {"name": "NAIP/Wyoming", "type": "ImageServer"}]}

# NAIP_KEY dict is deprecated in ApfoNaip class, it has since been switched to a
# more straightforward histical query, e.g. .../services/NAIP_Historical/WA_NAIP/ImageServer/...
NAIP_KEY = {'alabama': 'NAIP/Alabama', 'arizona': 'NAIP/Arizona', 'arkansas': 'NAIP/Arkansas',
            'california': 'NAIP/California_2016_60cm', 'colorado': 'NAIP/Colorado',
            'connecticut': 'NAIP/Connecticut_2016_60cm', 'delaware': 'NAIP/Delaware', 'florida': 'NAIP/Florida',
            'georgia': 'NAIP/Georgia', 'idaho': 'NAIP/Idaho', 'illinois': 'NAIP/Illinois',
            'indiana': 'NAIP/Indiana_2016_60cm', 'iowa': 'NAIP/Iowa', 'kansas': 'NAIP/Kansas',
            'kentucky': 'NAIP/Kentucky_2016_60cm', 'louisiana': 'NAIP/Louisiana', 'maine': 'NAIP/Maine_2015_1m',
            'maryland': 'NAIP/Maryland', 'massachusetts': 'NAIP/Massachusetts_2016_60cm',
            'michigan': 'NAIP/Michigan_2016_60cm', 'minnesota': 'NAIP/Minnesota',
            'mississippi': 'NAIP/Mississippi_2016_60cm', 'missouri': 'NAIP/Missouri_2016_60cm',
            'montana': 'NAIP/Montana_2015_1m', 'nebraska': 'NAIP/Nebraska_2016_60cm', 'nevada': 'NAIP/Nevada',
            'new_hampshire': 'NAIP/New_Hampshire_2016_60cm', 'new_jersey': 'NAIP/New_Jersey',
            'new_mexico': 'NAIP/New_Mexico_2016_1m', 'new_york': 'NAIP/New_York',
            'north_carolina': 'NAIP/North_Carolina_2016_1m', 'north_dakota': 'NAIP/North_Dakota', 'ohio': 'NAIP/Ohio',
            'oklahoma': 'NAIP/Oklahoma', 'oregon': 'NAIP/Oregon_2016_1m', 'pennsylvania': 'NAIP/Pennsylvania',
            'rhode_island': 'NAIP/Rhode_Island_2016_60cm', 'south_carolina': 'NAIP/South_Carolina',
            'south_dakota': 'NAIP/South_Dakota_2016_60cm', 'tennessee': 'NAIP/Tennessee_2016_60cm',
            'texas': 'NAIP/Texas_2016_1m', 'utah': 'NAIP/Utah_2016_1m', 'vermont': 'NAIP/Vermont_2016_60cm',
            'virginia': 'NAIP/Virginia_2016_1m', 'washington': 'NAIP/Washington',
            'west_virginia': 'NAIP/West_Virginia_2016_1m', 'wisconsin': 'NAIP/Wisconsin', 'wyoming': 'NAIP/Wyoming'}


def get_naip_key(state):
    return NAIP_KEY[state]


if __name__ == '__main__':
    home = os.path.expanduser('~')

# ========================= EOF ================================================================

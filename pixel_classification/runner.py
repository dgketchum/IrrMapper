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

home = os.path.expanduser('~')
root = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')

ID = os.path.join(root, 'ID')
MT = os.path.join(root, 'MT')
NV = os.path.join(root, 'NV')
OR = os.path.join(root, 'OR')
UT = os.path.join(root, 'UT')
WA = os.path.join(root, 'WA')


def build_compiled_feature_array():
    pass


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

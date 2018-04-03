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
import pickle
import itertools
import pkg_resources
from collections import OrderedDict
from datetime import datetime
from pandas import DataFrame, Series
from numpy import linspace, round, min, max
from numpy.random import shuffle

from fiona import open as fopen
from fiona import collection
from rasterio import open as rasopen
from shapely.geometry import shape, Polygon, Point, mapping
from shapely.ops import unary_union, cascaded_union

from pixel_prep.nlcd_map import map_nlcd_to_flu, nlcd_value
from sat_image.image import LandsatImage

WRS_2 = pkg_resources.resource_filename('spatial_data', 'wrs2_descending.shp')

# TODO: move all pixel prep functionality into one class!

# class PixelTrainingArray(LandsatImage):
#
#     def __init__(self, training_shapefile, year):

if __name__ == '__main__':
    home = os.path.expanduser('~')


# ========================= EOF ================================================================

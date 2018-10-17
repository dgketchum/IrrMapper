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

from numpy import linspace, max
from numpy.random import shuffle
from pandas import DataFrame
import fiona
from pyproj import Proj
from shapely.geometry import shape, Point
from shapely.ops import unary_union

OPEN_WATER = ['MT_Wetlands_Eastopen_water.shp',
              'WA_Wetlands_Westopen_water.shp',
              'CA_Wetlands_NorthCentralopen_water.shp',
              'CA_Wetlands_SouthCentralopen_water.shp',
              'WY_Wetlands_Eastopen_water.shp',
              'OR_Wetlands_Eastopen_water.shp',
              'NM_Wetlandsopen_water.shp',
              'CO_Wetlands_Westopen_water.shp',
              'ID_Wetlandsopen_water.shp',
              'AZ_Wetlandsopen_water.shp',
              'CO_Wetlands_Eastopen_water.shp',
              'MT_Wetlands_Westopen_water.shp',
              'WA_Wetlands_Eastopen_water.shp',
              'NV_Wetlands_Southopen_water.shp',
              'OR_Wetlands_Westopen_water.shp',
              'CA_Wetlands_Northopen_water.shp',
              'WY_Wetlands_Westopen_water.shp',
              'UT_Wetlandsopen_water.shp',
              'NV_Wetlands_Northopen_water.shp']

WETLAND = ['MT_Wetlands_Eastwetlands.shp',
           'WA_Wetlands_Westwetlands.shp',
           'CA_Wetlands_NorthCentralwetlands.shp',
           'CA_Wetlands_SouthCentralwetlands.shp',
           'WY_Wetlands_Eastwetlands.shp',
           'OR_Wetlands_Eastwetlands.shp',
           'NM_Wetlandswetlands.shp',
           'CO_Wetlands_Westwetlands.shp',
           'ID_Wetlandswetlands.shp',
           'AZ_Wetlandswetlands.shp',
           'CO_Wetlands_Eastwetlands.shp',
           'MT_Wetlands_Westwetlands.shp',
           'WA_Wetlands_Eastwetlands.shp',
           'NV_Wetlands_Southwetlands.shp',
           'OR_Wetlands_Westwetlands.shp',
           'CA_Wetlands_Northwetlands.shp',
           'WY_Wetlands_Westwetlands.shp',
           'UT_Wetlandswetlands.shp',
           'NV_Wetlands_Northwetlands.shp']

MT_SHP = ['West_Bench_Canal.shp',
          'East_Fork_Main_Canal_ab_Trout_Creek.shp',
          'Broadwater_Missouri_West_Side_Canal.shp',
          'Paradise_Valley_ID.shp',
          'Broadwater_Missouri_Canal.shp',
          'Huntley_Main_Diversion.shp',
          'Vigilante_Canal.shp',
          'LYIP_North_Dakota.shp',
          'Yellowstone_Main_Diversion.shp',
          'Ruby_River.shp',
          'Eldorado.shp',
          'Glasgow_ID.shp',
          'Fort_Belknap_Main_Diversion.shp',
          'Dodson_North_Canal_Diversion.shp',
          'Sun_River_project_Below_Pishkun.shp',
          'Fort_Shaw_Canal.shp',
          'Dodson_South_Div_To_Bowdoin.shp',
          'Floweree_and_Floweree_Hamilton.shp',
          'Marshall_Canal.shp']


class PointsRunspec(object):

    def __init__(self, root, **kwargs):
        self.root = root
        self.features = []
        self.object_id = 0
        self.year = None
        self.aea = Proj(
            '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs')
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE', 'YEAR'])

        if 'surface_water' in kwargs.keys():
            self.surface_water(kwargs['surface_water'])
        if 'wetlands' in kwargs.keys():
            self.surface_water(kwargs['wetlands'])
        if 'unirrigated' in kwargs.keys():
            self.surface_water(kwargs['unirrigated'])
        if 'forest' in kwargs.keys():
            self.surface_water(kwargs['forest'])

        self.irrigated(n)

    def surface_water(self, n):
        _files = [os.path.join(self.root, 'wetlands', x) for x in OPEN_WATER]
        areas = self.areas(_files)
        total_area = sum([x[1] for x in areas])
        samples = [(s, (a * n / total_area)) for s, a in areas]
        for s, n in samples:
            self.create_sample_points(n, s, code=1)

    def wetlands(self, n):
        pass

    def unirrigated(self, n):
        pass

    def forest(self, n):
        pass

    def irrigated(self, n):
        key = {'MT': 'Montana',
               'NV': 'Nevada',
               'OR': 'Oregon',
               'UT': 'Utah',
               'WA': 'Washington'}

    def areas(self, *shapes):
        totals = []
        for shp in shapes:
            area = self._shapefile_area(shp)
            totals.append((shp, area))
        return totals

    def _shapefile_area(self, shp):
        a = 0
        with fiona.open(shp, 'r') as src:
            print(shp)
            for feat in src:
                lon, lat = zip(*feat['coordinates'][0])
                x, y = self.aea(lon, lat)
                cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
                a += shape(cop).area
        return a

    def create_sample_points(self, n, shp, code):

        polygons = self._get_polygons(shp)
        instance_ct = 0

        if len(polygons) > n:
            areas = zip(polygons, [x.area for x in polygons])
            srt = sorted(areas, key=lambda x: x[1], reverse=True)
            polygons = [x for x, y in srt[:n]]

        polygons = unary_union(polygons)
        positive_area = sum([x.area for x in polygons])
        class_count = 0

        for i, poly in enumerate(polygons):
            if class_count >= n:
                break
            fractional_area = poly.area / positive_area
            required_points = max([1, fractional_area * n])
            x_range, y_range = self._random_points(poly.bounds, n)
            poly_pt_ct = 0

            for coord in zip(x_range, y_range):
                if Point(coord[0], coord[1]).within(poly):
                    self._add_entry(coord, val=code)
                    poly_pt_ct += 1
                    instance_ct += 1

                if instance_ct > n:
                    break

                if poly_pt_ct >= required_points:
                    break

            class_count += poly_pt_ct

    def _get_polygons(self, vector):
        with fiona.open(vector, 'r') as src:
            polys = []
            bad_geo_count = 0
            for feat in src:
                try:
                    geo = shape(feat['geometry'])
                    polys.append(geo)
                except AttributeError:
                    bad_geo_count += 1

        return polys

    def _random_points(self, coords, n):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * n)
        y_range = linspace(min_y, max_y, num=2 * n)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val,
                                                              'YEAR': self.year},
                                                             ignore_index=True)
        self.object_id += 1

    def save_sample_points(self):

        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10'), ('YEAR', 'int:10')]),
            'geometry': 'Point'}
        meta = self.tile_geometry.copy()
        meta['schema'] = points_schema

        with fopen(self.shapefile_path, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']),
                              ('POINT_TYPE', row['POINT_TYPE']),
                              ('YEAR', self.year)])

                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

if __name__ == '__main__':
    home = os.path.expanduser('~')
    gis = os.path.join(home, 'IrrigationGIS', 'EE_sample')
    prs = PointsRunspec(gis)
# ========================= EOF ====================================================================

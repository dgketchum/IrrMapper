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

import fiona
from numpy import linspace, max, arange
from numpy.random import shuffle
from pandas import DataFrame
from pyproj import Proj
from shapely.geometry import shape, Point, mapping

OPEN_WATER = [
    'MT_Wetlands_East_ow_1000.shp',
              'WA_Wetlands_West_ow_1000.shp',
              'CA_Wetlands_NorthCentral_ow_1000.shp',
              'CA_Wetlands_SouthCentral_ow_1000.shp',
              'WY_Wetlands_East_ow_1000.shp',
              'OR_Wetlands_East_ow_1000.shp',
              'NM_Wetlands_ow_1000.shp',
              'CO_Wetlands_West_ow_1000.shp',
              'ID_Wetlands_ow_1000.shp',
              'AZ_Wetlands_ow_1000.shp',
              'CO_Wetlands_East_ow_1000.shp',
              'MT_Wetlands_West_ow_1000.shp',
              'WA_Wetlands_East_ow_1000.shp',
              'NV_Wetlands_South_ow_1000.shp',
              'OR_Wetlands_West_ow_1000.shp',
              'CA_Wetlands_North_ow_1000.shp',
              'WY_Wetlands_West_ow_1000.shp',
              'UT_Wetlands_ow_1000.shp',
              'NV_Wetlands_North_ow_1000.shp'
]

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
        self.meta = None
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE', 'YEAR'])

        if 'surface_water' in kwargs.keys():
            self.surface_water(kwargs['surface_water'])
        if 'wetlands' in kwargs.keys():
            self.surface_water(kwargs['wetlands'])
        if 'unirrigated' in kwargs.keys():
            self.surface_water(kwargs['unirrigated'])
        if 'forest' in kwargs.keys():
            self.surface_water(kwargs['forest'])

        # self.irrigated(kwargs['irrigated'])

    def surface_water(self, n):
        _files = [os.path.join(self.root, 'open_water', x) for x in OPEN_WATER]
        areas = self.shapefile_area_count(_files)
        total_area = sum([x[1] for x in areas])
        samples = [(s, (a * n / total_area), ct) for s, a, ct in areas]
        for s, n, ct in samples:
            print(ct, ':', s)
            self.create_sample_points(n, s, code=1)

    def wetlands(self, n):
        pass

    def unirrigated(self, n):
        pass

    def forest(self, n):
        pass

    def irrigated(self, n):
        pass

    def shapefile_area_count(self, shapes):
        a = 0
        totals = []
        for shp in shapes:
            ct = 0
            with fiona.open(shp, 'r') as src:
                if not self.meta:
                    self.meta = src.meta
                print(shp)
                for feat in src:
                    if src.crs['units'] == 'm':
                        cop = {"type": "Polygon", "coordinates": [feat['geometry']['coordinates'][0]]}
                    else:
                        lon, lat = zip(*feat['geometry']['coordinates'][0])
                        x, y = self.aea(lon, lat)
                        cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
                    try:
                        a += shape(cop).area
                        ct += 1
                    except Exception:
                        pass
            totals.append((shp, a, ct))

        return totals

    def create_sample_points(self, n, shp, code):

        instance_ct = 0

        polygons = self._get_polygons(shp)
        positive_area = sum([x.area for x in polygons])
        for i, poly in enumerate(polygons):
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

    def save_sample_points(self, path):

        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10'), ('YEAR', 'int:10')]),
            'geometry': 'Point'}
        self.meta['schema'] = points_schema

        with fiona.open(path, 'w', **self.meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']),
                              ('POINT_TYPE', row['POINT_TYPE']),
                              ('YEAR', self.year)])

                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

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


if __name__ == '__main__':
    home = os.path.expanduser('~')
    gis = os.path.join(home, 'IrrigationGIS', 'EE_sample')
    prs = PointsRunspec(gis, **{'surface_water': 100})
    prs.save_sample_points(os.path.join(gis, 'sample_100.shp'))

# ========================= EOF ====================================================================

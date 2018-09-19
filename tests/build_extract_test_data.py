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
import shutil
import fiona
import rasterio
from rasterio.mask import mask
from fiona import crs
from pyproj import Proj, transform
from shapely.geometry import Polygon, mapping

from pixel_classification.runspec import MontanaTest


def make_test_dataset(coords_wsen, image_dir, new_test_dir, training):
    make_dir(new_test_dir)
    new_polygon_path = make_polygon(coords_wsen, new_test_dir)
    clip_training_vector(new_test_dir, coords_wsen, training)
    clip_images(image_dir, new_polygon_path, new_test_dir)
    return None


def make_dir(new_test_dir):
    if not os.path.isdir(new_test_dir):
        os.mkdir(new_test_dir)


def make_polygon(coordinates, test_data_dir):
    w, s, e, n = coordinates
    linear_ring = [(e, n), (w, n), (w, s), (e, s)]
    schema = {'geometry': 'Polygon',
              'properties': {'FID': 'int:10'}}
    args = {'mode': 'w', 'driver': 'ESRI Shapefile', 'schema': schema, 'crs': crs.from_epsg(4326)}
    new_polygon = os.path.join(test_data_dir, 'polygon.shp')
    with fiona.open(new_polygon, **args) as output:
        poly = Polygon(shell=linear_ring)
        prop = {'FID': 1}
        output.write({'geometry': mapping(poly), 'properties': prop})

    return new_polygon


def clip_training_vector(test_directory, coords, geography):
    w, s, e, n = coords
    for class_code, _dict in geography.attributes.items():

        with fiona.open(_dict['path']) as trn:
            clipped = trn.filter(bbox=(w, s, e, n))

            args = {'mode': 'w', 'driver': 'ESRI Shapefile',
                    'schema': trn.schema, 'crs': crs.from_epsg(4326)}
            clip_train_vector = _dict['path'].replace('.shp', '_clip.shp')
            training_clip = os.path.join(test_directory, os.path.basename(clip_train_vector))

            with fiona.open(training_clip, **args) as clip:
                for elem in clipped:
                    clip.write(elem)


def clip_images(image_dir, new_polygon, new_test_dir):
    with fiona.open(new_polygon) as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]

    first = True
    image_dirs = [x for x in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, x))]
    for image in image_dirs:
        for tif in os.listdir(os.path.join(image_dir, image)):

            in_tif = os.path.join(image_dir, image, tif)
            out_tif = os.path.join(new_test_dir, image, tif)

            if not os.path.isdir(os.path.dirname(out_tif)):
                os.mkdir(os.path.dirname(out_tif))

            if tif.endswith('MTL.txt'):
                shutil.copyfile(in_tif, out_tif)

            if tif.lower().endswith('.tif'):

                if first:
                    with rasterio.open(in_tif, 'r') as ras:
                        ref_system = ras.crs
                    init = ref_system['init']

                    projected_geoms = []
                    for tup in geoms[0]['coordinates'][0]:
                        projected_geoms.append(_geo_to_projected(tup[0], tup[1], init))
                    geoms[0]['coordinates'][0] = projected_geoms
                    first = False

                with rasterio.open(in_tif, 'r') as src:
                    out_image, out_transform = mask(src, geoms, crop=True)
                    out_meta = src.meta.copy()

                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})

                with rasterio.open(out_tif, 'w', **out_meta) as dst:
                    dst.write(out_image)
            else:
                pass


def _geo_to_projected(x, y, out_crs):
    in_crs = Proj(init='epsg:4326')
    out_crs = Proj(init=out_crs)
    x, y = transform(in_crs, out_crs, x, y)
    return x, y


def get_fiona(shapefile):
    lats = []
    lons = []
    with fiona.open(shapefile, 'r') as src:
        for feat in src:
            coords = feat['geometry']['coordinates'][0]
            [(lons.append(x), lats.append(y)) for x, y in coords]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    print('South extent: {}\nNorth extent: {}\nWest extent: {}\nEast extent: {}'.format(min_lat,
                                                                                        max_lat,
                                                                                        min_lon,
                                                                                        max_lon))



if __name__ == '__main__':
    home = os.path.expanduser('~')
    coords = -111.67, 47.17, -111.20, 47.48
    shp = os.path.join(home, 'IrrigationGIS', 'OE_Shapefiles_WGS', 'Broadwater_Missouri_Canal.shp')
    get_fiona(shp)
    # make_test_dataset(coords, _dir, test_dir, geo)
# ========================= EOF ====================================================================

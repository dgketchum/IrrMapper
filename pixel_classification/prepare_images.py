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
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
from numpy import mean, datetime64
from collections import OrderedDict
from datetime import datetime
from landsat.google_download import GoogleDownload
from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask
from sat_image.warped_vrt import warp_vrt
from met.thredds import GridMet, TopoWX
from bounds import RasterBounds, GeoBounds
from dem import AwsDem
from ssebop_app.image import get_image
from functools import partial
from pyproj import Proj, transform as pytransform
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform
from rasterio import open as rasopen, float32
from pixel_classification.crop_data_layer import CropDataLayer as Cdl
from pixel_classification.runspec import landsat_rasters, static_rasters, ancillary_rasters, mask_rasters
from sklearn.preprocessing import StandardScaler
from geopandas.geodataframe import GeoDataFrame

class ImageStack(object):
    """
    Prepare a stack of images from Landsat, terrain, etc. Save stack in identical geometry.
    """

    def __init__(self, satellite, path=None, row=None, lat=None, lon=None, root=None, max_cloud_pct=None, start=None, end=None, year=None, n_landsat=None):

        self.landsat_mapping = {'LT5': Landsat5, 'LE7': Landsat7, 'LC8': Landsat8}
        self.landsat_mapping_abv = {5: 'LT5', 7: 'LE7', 8: 'LC8'}

        self.sat = satellite
        self.sat_abv = self.landsat_mapping_abv[self.sat]
        self.sat_obj = self.landsat_mapping[self.sat_abv]

        self.path = path
        self.row = row
        self.lat = lat
        self.lon = lon
        self.year = year

        self.max_cloud = max_cloud_pct
        self.start = start
        self.end = end
        self.root = root

        self.profile = None
        self.dst_path_cloud = None
        self.dst_path_water = None
        self.landsat = None
        self.scenes = None
        self.image_dirs = None
        self.image_paths = None
        self.stack_features = None
        self.paths_map = None
        self.masks = None

        self.cdl_tif = None
        self.cdl_mask = None

        self.n = n_landsat

        self.ancillary_rasters = []
        self.exclude_rasters = []

        if year and not start and not end:
            self.start = '{}-05-01'.format(self.year)
            self.end = '{}-10-15'.format(self.year)

    def build_training(self):
        self.get_landsat(fmask=True)
        self.profile = self.landsat.rasterio_geometry
        self.get_precip()
        self.get_et()
        self.get_terrain()
        self.get_cdl()
        self.paths_map, self.masks = self._order_images()

    def build_evaluating(self):
        self.get_landsat(fmask=False)
        self.profile = self.landsat.rasterio_geometry
        #self.get_et()
        self.get_precip()
        self.get_terrain()
        self.get_cdl()
        self.paths_map, self.masks = self._order_images() # paths map is just path-> location
        # in filesystem.

    def get_cdl(self):
        """download cdl and make a mask, save to the
        root directory with filename cdl_mask.tif.
        The cdl is reprojected here.
        """
        self.cdl_mask = os.path.join(self.root, 'cdl_mask.tif')
        if not os.path.isfile(self.cdl_mask):
            print('get {}'.format(self.cdl_mask))
            polygon = self.landsat.get_tile_geometry()
            cdl = Cdl(year=self.year, target_profile=self.landsat.profile)
            cdl.get_mask(clip_geometry=polygon, out_file=self.cdl_mask)
        else:
            print('{} exists'.format(self.cdl_mask))
            self.exclude_rasters.append(self.cdl_mask)

    def get_landsat(self, fmask=False):
        """Download from internet and select scenes from n_landsat
        g.download() then saves the selected scenes into
        the root directory.
        """
        if self.lat is None:
            g = GoogleDownload(self.start, self.end, self.sat, path=self.path, row=self.row,
                               output_path=self.root, max_cloud_percent=self.max_cloud)
        else:
            g = GoogleDownload(self.start, self.end, self.sat, latitude=self.lat, longitude=self.lon,
                               output_path=self.root, max_cloud_percent=self.max_cloud)
            self.path = g.p
            self.row = g.r

        g.select_scenes(self.n)
        self.scenes = g.selected_scenes
        g.download(list_type='selected')

        self.image_dirs = [x[0] for x in os.walk(self.root) if
                           os.path.basename(x[0])[:3] in self.landsat_mapping.keys()]

        self._get_geography()
        if fmask:
            [self._make_fmask(d) for d in self.image_dirs]

    def get_precip(self):
        poly_in = self.landsat.get_tile_geometry()
        poly_in = Polygon(poly_in[0]['coordinates'][0])
        project = partial(
                pytransform, 
                Proj(self.profile['crs']),
                Proj(init='epsg:32612'))
        for_bounds = partial(
                pytransform, 
                Proj(self.profile['crs']),
                Proj(init='epsg:4326'))
        dates = self.scenes['DATE_ACQUIRED'].values
        # Change the coordinate system
        # Ask david
        poly = transform(project, poly_in)
        poly_bounds = transform(for_bounds, poly_in)
        poly = Polygon(poly.exterior.coords)
        from rasterio.crs import CRS
        geometry = [mapping(poly)] 
        geometry[0]['crs'] = CRS({'init':'epsg:32612'})
        feat = {'type': 'Polygon', 'coordinates': list(poly.exterior.coords)}
        bounds = poly.bounds
        print(bounds)
        bounds = (bounds[2], bounds[1], bounds[0], bounds[3])
        bounds = (-124.84, -66.88, 24.89, 49.38) # bbox of usa for sanity check
        bounds = poly_bounds.bounds
        for date in dates:
            d = datetime.utcfromtimestamp(date.tolist()/1e9) # convert to a nicer format.
            bds = GeoBounds(wsen=bounds)
            gm = GridMet(variable='pr', clip_feature=geometry,
                    bbox=bds, target_profile=self.profile, date=d)
            out = gm.get_data_subset()
            outfile = os.path.join(self.root, 'GridMet{}.tif'.format(date))
            gm.save_raster(out, self.landsat.rasterio_geometry, outfile)

    def get_terrain(self):
        """
        Get digital elevation maps from amazon web services
        save in the project root directory with filenames enumerated
        in the next three lines.

        """

        slope_name = os.path.join(self.root, 'slope.tif')
        aspect_name = os.path.join(self.root, 'aspect.tif')
        dif_elev = os.path.join(self.root, 'elevation_diff.tif')

        check = [os.path.isfile(x) for x in [slope_name, aspect_name, dif_elev]]

        if False in check:
            polygon = self.landsat.get_tile_geometry()
            bb = RasterBounds(affine_transform=self.profile['transform'],
                              profile=self.profile, latlon=True)
            dem = AwsDem(zoom=10, target_profile=self.profile, bounds=bb,
                         clip_object=polygon)

            dem.terrain(attribute='slope',
                        out_file=slope_name, save_and_return=True)
            dem.terrain(attribute='aspect',
                        out_file=aspect_name, save_and_return=True)
            elev = dem.terrain(attribute='elevation')
            elev = elev - mean(elev)
            dem.save(elev, geometry=dem.target_profile, output_filename=dif_elev)

    def get_et(self):
        for i, d in enumerate(self.image_dirs):
            l = self.landsat_mapping[self.sat_abv](d)
            _id = l.landsat_scene_id
            get_image(image_dir=d, parent_dir=self.root, image_exists=True, image_id=_id,
                      satellite=self.sat, path=self.path, row=self.row, image_date=l.date_acquired,
                      landsat_object=self.landsat, overwrite=False)

    def warp_vrt(self):
        warp_vrt(self.root, delete_extra=False, use_band_map=False, remove_bqa=True)

    def _get_geography(self):

        master = self.image_dirs[0]
        self.landsat = self.landsat_mapping[self.sat_abv](master)

    def _make_fmask(self, image_dir):
        s = os.path.basename(image_dir)
        self.dst_path_cloud = os.path.join(image_dir, '{}_cloud_fmask.tif'.format(s))
        self.dst_path_water = os.path.join(image_dir, '{}_water_fmask.tif'.format(s))

        if os.path.isfile(self.dst_path_cloud) and os.path.isfile(self.dst_path_water):
            print('{} and {} exist for {}'.format(os.path.basename(self.dst_path_cloud),
                                                  os.path.basename(self.dst_path_water),
                                                  image_dir))

        else:
            print('fmask for {}'.format(image_dir))
            lst_image = self.landsat_mapping[self.sat_abv](image_dir)

            f = Fmask(lst_image)

            c, shadow, water = f.cloud_mask()
            cloud = c | shadow

            f.save_array(cloud, self.dst_path_cloud)
            f.save_array(water, self.dst_path_water)

    def _organize_directory(self):
        dst_dir = os.path.join(self.root, str(self.path), str(self.row),
                               str(self.year))
        if not os.path.isdir(dst_dir):

            try:
                os.makedirs(dst_dir)
                print('Made {}'.format(dst_dir))

            except Exception:
                pass

        return dst_dir

    def _order_images(self):
        band_dct = OrderedDict()
        mask_dct = OrderedDict()

        if not self.image_dirs:
            raise NotImplementedError('must build stack with "build_all" before listing rasters')

        dates = self.scenes['DATE_ACQUIRED'].values
        scenes = self.scenes['SCENE_ID'].values
        s = datetime64('{}-01-01'.format(self.year))
        for d in dates:
            try:
                assert d > s
            except AssertionError:
                print('Scene dates appear to not increase monotonically')
                raise NotImplementedError
            s = d

        for sc in scenes:
            paths = os.listdir(os.path.join(self.root, sc))
            b = [os.path.join(self.root, sc, x) for x in paths if x.endswith(landsat_rasters()[self.sat])]
            a = [os.path.join(self.root, sc, x) for x in paths if x.endswith(ancillary_rasters())]
            bands = a + b
            bands.sort()
            for p in bands:
                band_dct[os.path.basename(p).split('.')[0]] = p
                self._normalize_and_save_image(p)

            masks = [os.path.join(self.root, sc, x) for x in paths if x.endswith(mask_rasters())]
            for m in masks:
                mask_dct[os.path.basename(m).split('.')[0]] = m

        dir_list = os.listdir(self.root)
        files = [x for x in dir_list if os.path.isfile(os.path.join(self.root, x))]
        static_files = [x for x in files if x.endswith(static_rasters())]
        for st in static_files:
            band_dct[os.path.basename(st).split('.')[0]] = os.path.join(self.root, st)
            self._normalize_and_save_image(os.path.join(self.root, st))

        return band_dct, mask_dct

    @staticmethod
    def _normalize_and_save_image(fname):
        norm = True
        with rasopen(fname, 'r') as rsrc:
            if "normalized" in rsrc.tags():
                return
            else:    
                rass_arr = rsrc.read()
                rass_arr = rass_arr.astype(float32)
                profile = rsrc.profile
                profile.update(dtype=float32)
                rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
                scaler = StandardScaler() # z-normalization
                scaler.fit(rass_arr)
                rass_arr = scaler.transform(rass_arr)
                with rasopen(fname, 'w', **profile) as dst:
                    dst.write(rass_arr, 1)
                    print("Normalized", fname)
                    dst.update_tags(normalized=True)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================

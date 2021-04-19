import ee

ee.Initialize()
import time
import os
import json
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ee_image.collection import InterpolatedCollection, MeanCollection
from ee_image.collection import get_target_dates, get_target_bands
from data_extract.bucket import get_bucket_contents

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'ta_data'

POINTS = 'users/dgketchum/grids/grid_pts'
GRID = 'users/dgketchum/grids/grid'

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

LC8_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']
LC7_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6']
LC5_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6']
STD_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']


def create_class_labels(year_, roi_):
    irrigated = 'users/dgketchum/training_polygons/irrigated'
    fallow = 'users/dgketchum/training_polygons/fallow'
    dryland = 'users/dgketchum/training_polygons/dryland'
    uncultivated = 'users/dgketchum/training_polygons/uncultivated'
    wetlands = 'users/dgketchum/training_polygons/wetlands'

    gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
    water = gsw.select('occurrence').gt(5).unmask(0)
    dataset = ee.Image('USGS/NLCD/NLCD2016')
    landcover = dataset.select('landcover')
    mask = landcover.lt(24)
    imperv = mask.updateMask(landcover.gt(24)).updateMask(water.Not()).unmask(1)
    mask = imperv.mask(imperv.gt(0)).add(3)

    class_labels = ee.Image(mask).byte()
    irrigated = ee.FeatureCollection(irrigated).filter(ee.Filter.eq("YEAR", year_)).filterBounds(roi_)
    fallow = ee.FeatureCollection(fallow).filter(ee.Filter.eq("YEAR", year_)).filterBounds(roi_)
    dryland = ee.FeatureCollection(dryland).merge(fallow).filterBounds(roi_)
    uncultivated = ee.FeatureCollection(uncultivated).merge(wetlands).filterBounds(roi_)

    class_labels = class_labels.paint(uncultivated, 3)
    class_labels = class_labels.paint(dryland, 2)
    class_labels = class_labels.paint(irrigated, 1)
    class_labels = class_labels.updateMask(class_labels).unmask(water)
    label = class_labels.rename('irr')
    return label


def get_ancillary(year):
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select(['elevation', 'slope', 'aspect']) \
        .resample('bilinear').rename(['elv', 'slp', 'asp'])

    if 2007 < year < 2017:
        cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filter(ee.Filter.date('{}-01-01'.format(year), '{}-12-31'.format(year))) \
            .first().select(['cropland', 'confidence']).rename(['cdl', 'cconf'])
    else:
        cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filter(ee.Filter.date('2017-01-01', '2017-12-31')) \
            .first().select(['cropland', 'confidence']).rename(['cdl', 'cconf'])

    return terrain, cdl


def get_means_stack(yr, s, e, interval, mask, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
    model_obj = MeanCollection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        mask=mask,
        geometry=geo_,
        interval=interval,
        cloud_cover_max=60)

    means, names = model_obj.get_means()

    return means, names


def get_sr_stack(yr, s, e, interval, mask, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
    target_interval = interval
    interp_days = 64

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = InterpolatedCollection(
        collections=COLLECTIONS,
        start_date=s,
        end_date=e,
        mask=mask,
        geometry=geo_,
        cloud_cover_max=60)

    variables_ = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

    interpolated = model_obj.interpolate(variables=variables_,
                                         interp_days=interp_days,
                                         dates=target_dates)

    target_bands, target_rename = get_target_bands(s, e, interval_=target_interval, vars=variables_)
    interp = interpolated.sort('system:time_start').toBands().rename(target_rename)
    return interp, target_rename


def extract_by_point(year, grid_fid=1440, point_fids=None, cloud_mask=True,
                     split='train', type='means'):
    if cloud_mask:
        cloud = 'cm'
    else:
        cloud = 'nm'

    grid = ee.FeatureCollection(GRID)
    roi = grid.filter(ee.Filter.eq('FID', grid_fid)).geometry()
    points_fc = ee.FeatureCollection(POINTS)

    s, e, interval_ = 1, 1, 30
    if type == 'means':
        image_stack, features = get_means_stack(year, s, e, interval_, cloud_mask, roi)
    elif type == 'interpolate':
        image_stack, features = get_sr_stack(year, s, e, interval_, cloud_mask, roi)

    irr = create_class_labels(year, roi)
    terrain_, cdl_ = get_ancillary(year)
    coords_ = image_stack.pixelLonLat().rename(['lon', 'lat'])
    image_stack = ee.Image.cat([image_stack, terrain_, coords_, cdl_, irr]).float()
    features = features + ['elv', 'slp', 'asp', 'lon', 'lat', 'cdl', 'cconf', 'irr']

    projection = ee.Projection('EPSG:5070')
    image_stack = image_stack.reproject(projection, None, 30)
    data_stack = image_stack.neighborhoodToArray(KERNEL)

    ct = 0
    n_features = len(point_fids)
    geometry_sample = ee.ImageCollection([])
    for loc in point_fids:
        point = points_fc.filter(ee.Filter.eq('FID', int(loc)))
        sample = data_stack.sample(
            region=point.geometry(),
            scale=30,
            tileScale=16,
            dropNulls=False)
        geometry_sample = geometry_sample.merge(sample)
        ct += 1
    out_filename = '{}_{}_{}_{}'.format(split, str(year), cloud, grid_fid)
    task = ee.batch.Export.table.toCloudStorage(
        collection=geometry_sample,
        bucket=GS_BUCKET,
        description=out_filename,
        fileNamePrefix=out_filename,
        fileFormat='TFRecord',
        selectors=features)

    try:
        task.start()
    except ee.ee_exception.EEException:
        print('waiting 50 minutes to export {} {}'.format(out_filename, ct))
        time.sleep(3000)
        task.start()

    print('exported {} {}, {} of {} features'.format(grid_fid, year, ct, n_features))


def run_extract_irr_points(input_json, overwrite=False):
    exported = None
    if not overwrite:
        # contents = get_bucket_contents('ts_data')[0]['nm_12']
        contents = get_bucket_contents('ta_data')[0]
        exported = [x[0].split('.')[0] for x in contents]

    with open(input_json) as j:
        grids = json.loads(j.read())

    shard_ct = {'train': 0, 'test': 0, 'valid': 0}
    for gfid, dct in grids.items():
        gfid = int(gfid)
        years = [v[-1] for k, v in dct.items()]
        years = set([i for s in years for i in s])
        pfids = [(y, [k for k, v in dct.items() if y in v[-1]]) for y in years]
        for y, pf in pfids:
            split = dct[pf[0]][0]
            shard_ct[split] += len(pf)
            record = '{}_{}_cm_{}'.format(split, y, gfid)
            if not overwrite:
                if record in exported:
                    print('{} exists'.format(record))
                    continue
            extract_by_point(year=y, grid_fid=gfid, point_fids=pf,
                             cloud_mask=True, split=split, type='means')
    print(shard_ct)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    alt_home = os.path.join(home, 'data')
    if os.path.isdir(alt_home):
        home = alt_home
    _json = os.path.join(os.path.dirname(__file__), 'master_shards.json')
    run_extract_irr_points(_json, overwrite=True)

# =====================================================================================================================

import ee

ee.Initialize()
import time
import os
import json

from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ee_image.collection import get_target_dates, Collection, get_target_bands

KERNEL_SIZE = 256
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'ts_data'

POINTS = 'users/dgketchum/grids/grid_pts'
GRID = 'users/dgketchum/grids/grid'

COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR',
               'LANDSAT/LE07/C01/T1_SR',
               'LANDSAT/LT05/C01/T1_SR']

CLASSES = ['uncultivated', 'dryland', 'fallow', 'irrigated']

LC8_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']
LC7_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6']
LC5_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6']
STD_NAMES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']


def ls8mask(img):
    sr_bands = img.select('B2', 'B3', 'B4', 'B5', 'B6', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_mult = img_masked.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


def preprocess_data(year):
    l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').select(LC8_BANDS, STD_NAMES)
    l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').select(LC7_BANDS, STD_NAMES)
    l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').select(LC5_BANDS, STD_NAMES)
    l5l7l8 = ee.ImageCollection(l7.merge(l8).merge(l5))

    return temporalCollection(l5l7l8, ee.Date('{}-01-01'.format(year)), 10, 36, 'days')


def temporalCollection(collection, start, count, interval, units):
    sequence = ee.List.sequence(0, ee.Number(count).subtract(1))
    originalStartDate = ee.Date(start)

    def filt(i):
        startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units)
        endDate = originalStartDate.advance(
            ee.Number(interval).multiply(ee.Number(i).add(1)), units)
        return collection.filterDate(startDate, endDate).reduce(ee.Reducer.mean())

    return ee.ImageCollection(sequence.map(filt))


def class_codes():
    return {'irrigated': 2,
            'fallow': 3,
            'dryland': 4,
            'uncultivated': 5}


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


def get_sr_stack(yr, s, e, interval, mask, geo_):
    s = datetime(yr, s, 1)
    e = datetime(yr + 1, e, 1)
    target_interval = interval
    interp_days = 64

    target_dates = get_target_dates(s, e, interval_=target_interval)

    model_obj = Collection(
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


def extract_by_point(year, grid_fid=1440, point_fids=None, cloud_mask=False, split='train'):

    if cloud_mask:
        cloud = 'cm'
    else:
        cloud = 'nm'

    grid = ee.FeatureCollection(GRID)
    roi = grid.filter(ee.Filter.eq('FID', grid_fid)).geometry()
    points_fc = ee.FeatureCollection(POINTS)

    s, e, interval_ = 1, 1, 30
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
    geometry_sample = None
    for loc in point_fids:
        point = points_fc.filter(ee.Filter.eq('FID', int(loc)))
        geometry_sample = ee.ImageCollection([])

        sample = data_stack.sample(
            region=point.geometry(),
            scale=30,
            numPixels=1,
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


def run_extract_irr_points(input_json):

    with open(input_json) as j:
        grids = json.loads(j.read())

    for pfid, dct in grids.items():
        pfid = int(pfid)
        years = [v[-1] for k, v in dct.items()]
        years = set([i for s in years for i in s])
        pfids = [(y, [k for k, v in dct.items() if y in v[-1]]) for y in years]
        for y, pf in pfids:
            split = dct[pf[0]][0]
            extract_by_point(year=y, grid_fid=pfid, point_fids=pf, cloud_mask=True, split=split)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    alt_home = os.path.join(home, 'data')
    if os.path.isdir(alt_home):
        home = alt_home
    _json = '/home/dgketchum/PycharmProjects/EEMapper/map/data/master_shards.json'
    run_extract_irr_points(_json)

# =====================================================================================================================

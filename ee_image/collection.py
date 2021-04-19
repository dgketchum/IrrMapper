import os
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY, MONTHLY
import ee

from ee_image.image import Image
import ee_image.inerpolate as interp


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated

    https://stevenloria.com/lazy-properties/
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class InterpolatedCollection:
    """"""

    def __init__(
            self,
            collections,
            start_date,
            end_date,
            geometry,
            variables=None,
            mask=True,
            cloud_cover_max=70):

        self.collections = collections
        self.mask = mask
        self.start_date = start_date
        self.end_date = end_date
        self.start_str = self.start_date.strftime('%Y-%m-%d')
        self.end_str = self.end_date.strftime('%Y-%m-%d')

        self.variables = variables
        self.geometry = geometry
        self.cloud_cover_max = cloud_cover_max
        self._interp_vars = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

        # If collections is a string, place in a list
        if type(self.collections) is str:
            self.collections = [self.collections]

    def _build(self, interp_vars):
        """Build a merged model variable image collection

        Parameters
        ----------
        variables : list
            Set a variable list that is different than the class variable list.
        start_date : str, optional
            Set a start_date that is different than the class start_date.
            This is needed when defining the scene collection to have extra
            images for interpolation.
        end_date : str, optional
            Set an exclusive end_date that is different than the class end_date.

        Returns
        -------
        ee.ImageCollection

        Raises
        ------
        ValueError if collection IDs are invalid.
        ValueError if variables is not set here and in class init.

        """

        # Build the variable image collection
        variable_coll = ee.ImageCollection([])
        for coll_id in self.collections:
            input_coll = ee.ImageCollection(coll_id) \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(self.geometry) \
                .filterMetadata('CLOUD_COVER_LAND', 'less_than',
                                self.cloud_cover_max)

            # Time filters are to remove bad (L5) and pre-op (L8) images
            if 'LT05' in coll_id:
                input_coll = input_coll.filter(ee.Filter.lt(
                    'system:time_start', ee.Date('2011-12-31').millis()))
            elif 'LC08' in coll_id:
                input_coll = input_coll.filter(ee.Filter.gt(
                    'system:time_start', ee.Date('2013-03-24').millis()))

            def compute_lsr(image):
                model_obj = Image.from_landsat_c1_sr(
                    sr_image=ee.Image(image), mask=self.mask)
                return model_obj.calculate(interp_vars)

            variable_coll = variable_coll.merge(
                ee.ImageCollection(input_coll.map(compute_lsr)))

        return variable_coll

    def scenes(self, variables):
        interp_vars = [band for band in variables]
        interp_vars.append('time')
        scene_coll = self._build(interp_vars)
        return ee.ImageCollection(scene_coll).select(variables)

    def interpolate(self, variables, interp_days=32, dates=None):

        ref_et = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
            .filter(ee.Filter.inList("system:time_start", dates)) \
            .select(['etr'], ['et_reference'])

        interp_vars = [band for band in variables]

        # Count will be determined using the aggregate_coll image masks
        if 'count' in variables:
            interp_vars.append('mask')

        interp_vars.append('time')

        # Build initial scene image collection
        scene_coll = self._build(interp_vars)
        # For count, compute the composite/mosaic image for the mask band only
        if 'count' in variables:
            aggregate_coll = interp.aggregate_to_daily(
                image_coll=scene_coll.select(['mask']),
                start_date=self.start_date, end_date=self.end_date)

            aggregate_coll = aggregate_coll.merge(
                ee.Image.constant(0).rename(['mask']).set({'system:time_start': ee.Date(self.start_str).millis()}))

        if 'mask' in interp_vars:
            interp_vars.remove('mask')

        # Interpolate to a daily time step
        daily_coll = interp.daily(
            target_coll=ref_et,
            source_coll=scene_coll.select(interp_vars),
            interp_days=interp_days,
            use_joins=True)

        interp_properties = {
            'cloud_cover_max': self.cloud_cover_max,
            'collections': ', '.join(self.collections),
            'interp_days': interp_days,
            'interp_method': 'linear',
            'model_name': 'IrrMapper'}

        def aggregate_image(agg_start_date, agg_end_date, date_format):

            image_list = []

            for var in variables:
                # Compute average of variable over the aggregation period
                img = daily_coll \
                    .filterDate(agg_start_date, agg_end_date) \
                    .mean().select([var]).float()

                image_list.append(img)

            if 'count' in variables:
                count_img = aggregate_coll \
                    .filterDate(agg_start_date, agg_end_date) \
                    .select(['mask']).count().rename('count').uint8()
                image_list.append(count_img)

            agg_image = ee.Image(image_list) \
                .set(interp_properties) \
                .set({'system:index': ee.Date(agg_start_date).format(date_format),
                      'system:time_start': ee.Date(agg_start_date).millis(),
                      })

            return agg_image

        def aggregate_daily(daily_img):
            agg_start_date = ee.Date(daily_img.get('system:time_start'))
            return aggregate_image(
                agg_start_date=agg_start_date,
                agg_end_date=ee.Date(agg_start_date).advance(1, 'day'),
                date_format='YYYYMMdd')

        interp_collection = ee.ImageCollection(daily_coll.map(aggregate_daily)).select(variables)
        return interp_collection

    def get_image_ids(self):
        """Return image IDs of the input images

        Returns
        -------
        list

        Notes
        -----
        This image list is based on the collection start and end dates and may
        not include all of the images used for interpolation.

        """
        return sorted(list(self._build().aggregate_array('image_id').getInfo()))


class MeanCollection:
    """"""

    def __init__(
            self,
            collections,
            start_date,
            end_date,
            geometry,
            interval,
            variables=None,
            mask=True,
            cloud_cover_max=70):
        self.collections = collections
        self.mask = mask
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.start_str = self.start_date.strftime('%Y-%m-%d')
        self.end_str = self.end_date.strftime('%Y-%m-%d')

        self.variables = variables
        self.geometry = geometry
        self.cloud_cover_max = cloud_cover_max
        self._interp_vars = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir']

        # If collections is a string, place in a list
        if type(self.collections) is str:
            self.collections = [self.collections]

    def get_means(self):
        d_times = [(d - timedelta(days=30), d + timedelta(days=30)) for d in rrule(dtstart=self.start_date,
                                                                                   until=self.end_date,
                                                                                   freq=MONTHLY)]
        d_strings = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d')) for x, y in d_times]

        dates = [(i, d, s, e) for i, (d, (s, e)) in enumerate(zip(d_times, d_strings))]

        first = True
        names = []
        for i, d, s, e in dates:
            bands, name_list = landsat_means(d[1].year, s, e, self.geometry, i)
            if first:
                input_bands = bands
                first = False
            else:
                input_bands = input_bands.addBands(bands)
            names.append(name_list)
        names = [x for sublist in names for x in sublist]
        return input_bands, names


def ls57mask(img):
    sr_bands = img.select('B1', 'B2', 'B3', 'B4', 'B5', 'B7')
    mask_sat = sr_bands.neq(20000)
    img_nsat = sr_bands.updateMask(mask_sat)
    mask1 = img.select('pixel_qa').bitwiseAnd(8).eq(0)
    mask2 = img.select('pixel_qa').bitwiseAnd(32).eq(0)
    mask_p = mask1.And(mask2)
    img_masked = img_nsat.updateMask(mask_p)
    mask_sel = img_masked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7'], ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    mask_mult = mask_sel.multiply(0.0001).copyProperties(img, ['system:time_start'])
    return mask_mult


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


def ls5_edge_removal(lsImage):
    inner_buffer = lsImage.geometry().buffer(-3000)
    buffer = lsImage.clip(inner_buffer)
    return buffer


def landsat_masked(yr, roi):
    start = '{}-01-01'.format(yr)
    end_date = '{}-01-01'.format(yr + 1)

    l5_coll = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls5_edge_removal).map(ls57mask)
    l7_coll = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls57mask)
    l8_coll = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(
        roi).filterDate(start, end_date).map(ls8mask)

    lsSR_masked = ee.ImageCollection(l7_coll.merge(l8_coll).merge(l5_coll))
    return lsSR_masked


def landsat_means(year, start, end, roi, append_name):
    lsSR_masked = landsat_masked(year, roi)
    names = ['B2_{}'.format(append_name),
             'B3_{}'.format(append_name),
             'B4_{}'.format(append_name),
             'B5_{}'.format(append_name),
             'B6_{}'.format(append_name),
             'B7_{}'.format(append_name)]
    bands_means = ee.Image(lsSR_masked.filterDate(start, end).map(
        lambda x: x.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], names)).mean())

    return bands_means, names


def get_target_dates(s, e, interval_=15):
    d_times = [(d, d + timedelta(days=1)) for d in rrule(dtstart=s, until=e, interval=interval_, freq=DAILY)]
    d_strings = [(x.strftime('%Y-%m-%d'), y.strftime('%Y-%m-%d')) for x, y in d_times]
    gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    images = [gm.filterDate(s, e).first().getInfo()['properties']['system:time_start'] for s, e in d_strings]
    return images


def get_target_bands(s, e, interval_=15, vars=None):
    d_times = [d for d in rrule(dtstart=s, until=e, interval=interval_, freq=DAILY)]
    d_strings = [x.strftime('%Y%m%d') for x in d_times]
    ints_ = [x for x in range(len(d_strings))]

    collection_bands = [['{}_{}'.format(d, b) for b in vars] for d in d_strings]
    collection_bands = [item for sublist in collection_bands for item in sublist]

    rename_bands = [['{}_{}'.format(b, d) for b in vars] for d in ints_]
    rename_bands = [item for sublist in rename_bands for item in sublist]

    return collection_bands, rename_bands


if __name__ == '__main__':
    pass
# =======================================================================================

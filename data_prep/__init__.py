# dates are generic, dates of each year as below, but data is from many years
# the year of the data is not used in training, just date position
DATES = {0: '19860101',
         1: '19860131',
         2: '19860302',
         3: '19860401',
         4: '19860501',
         5: '19860531',
         6: '19860630',
         7: '19860730',
         8: '19860829',
         9: '19860928',
         10: '19861028',
         11: '19861127',
         12: '19861227'}

# see feature_spec.py for dict of bands, lat , lon, elev, label
CHANNELS = 7
BANDS = 91
TERRAIN = 5

SEQUENCE_LEN = len(DATES.keys())


FEATURES = ['blue_0', 'green_0', 'red_0', 'nir_0', 'swir1_0', 'swir2_0', 'tir_0', 'blue_1', 'green_1', 'red_1', 'nir_1',
            'swir1_1', 'swir2_1', 'tir_1', 'blue_2', 'green_2', 'red_2', 'nir_2', 'swir1_2', 'swir2_2', 'tir_2',
            'blue_3', 'green_3', 'red_3', 'nir_3', 'swir1_3', 'swir2_3', 'tir_3', 'blue_4', 'green_4', 'red_4', 'nir_4',
            'swir1_4', 'swir2_4', 'tir_4', 'blue_5', 'green_5', 'red_5', 'nir_5', 'swir1_5', 'swir2_5', 'tir_5',
            'blue_6', 'green_6', 'red_6', 'nir_6', 'swir1_6', 'swir2_6', 'tir_6', 'blue_7', 'green_7', 'red_7', 'nir_7',
            'swir1_7', 'swir2_7', 'tir_7', 'blue_8', 'green_8', 'red_8', 'nir_8', 'swir1_8', 'swir2_8', 'tir_8',
            'blue_9', 'green_9', 'red_9', 'nir_9', 'swir1_9', 'swir2_9', 'tir_9', 'blue_10', 'green_10', 'red_10',
            'nir_10', 'swir1_10', 'swir2_10', 'tir_10', 'blue_11', 'green_11', 'red_11', 'nir_11', 'swir1_11',
            'swir2_11', 'tir_11', 'blue_12', 'green_12', 'red_12', 'nir_12', 'swir1_12', 'swir2_12', 'tir_12', 'lat',
            'lon', 'elev']


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

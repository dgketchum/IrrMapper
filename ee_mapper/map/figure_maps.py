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
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import rasterio


def plot_map(shp, raster):
    gdata = gdal.Open(raster)
    geo = gdata.GetGeoTransform()
    data = gdata.ReadAsArray()

    xres = geo[1]
    yres = geo[5]

    # A good LCC projection for USA plots
    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

    # This just plots the shapefile -- it has already been clipped
    m.readshapefile(shp, 'states', drawbounds=True, color='0.3')

    xmin = geo[0] + xres * 0.5
    xmax = geo[0] + (xres * gdata.RasterXSize) - xres * 0.5
    ymin = geo[3] + (yres * gdata.RasterYSize) + yres * 0.5
    ymax = geo[3] - yres * 0.5

    x, y = np.mgrid[xmin:xmax + xres:xres, ymax + yres:ymin:yres]
    x, y = m(x, y)

    cmap = plt.cm.gist_rainbow
    cmap.set_under('1.0')
    cmap.set_bad('0.8')

    im = m.pcolormesh(x, y, data.T, cmap=cmap, vmin=0, vmax=100)

    cb = plt.colorbar(orientation='vertical', fraction=0.10, shrink=0.7)
    plt.title('August Precip (mm)')
    plt.show()


def resample_sum(raster):

    with rasterio.open(raster, 'r') as src:
        array_in = src.read()
        

        # arrayin, areas, rowsin, colsin, rowsout, colsout, curtargr, curtargc):

        firstinr = float(curtargr) * (float(rowsin) / float(rowsout))
        firstinrw = 1 - firstinr + int(firstinr)
        firstinr = int(firstinr)

        lastinr = (float(curtargr) + 1.) * (float(rowsin) / float(rowsout))
        lastinrw = lastinr - int(lastinr)
        lastinr = int(lastinr)

        firstinc = float(curtargc) * float((colsin) / float(colsout))
        firstincw = 1 - firstinc + int(firstinc)
        firstinc = int(firstinc)

        lastinc = (float(curtargc) + 1.) * (float(colsin) / float(colsout))
        lastincw = lastinc - int(lastinc)
        lastinc = int(lastinc)

        rowslength = int(lastinr - firstinr)
        if lastinrw > 0.:
            rowslength += 1
            if rowslength == 1:
                rowweights = [lastinrw + firstinrw - 1.]
            else:
                rowweights = [firstinrw]
                rowweights.extend(np.ones(rowslength - 2).tolist())
                rowweights.extend([lastinrw])
        else:
            lastinr -= 1
            if rowslength == 1:
                rowweights = [firstinrw]
            else:
                rowweights = [firstinrw]
                rowweights.extend(np.ones(rowslength - 1).tolist())

        rowinds = np.linspace(firstinr, lastinr, rowslength, dtype='uint32')

        colslength = int(lastinc - firstinc)
        if lastincw > 0.:
            colslength += 1
            if colslength == 1:
                colweights = [lastincw + firstincw - 1.]
            else:
                colweights = [firstincw]
                colweights.extend(np.ones(colslength - 2).tolist())
                colweights.extend([lastincw])
        else:
            lastinc -= 1
            if colslength == 1:
                colweights = [firstincw]
            else:
                colweights = [firstincw]
                colweights.extend(np.ones(colslength - 1).tolist())

        colinds = np.linspace(firstinc, lastinc, colslength, dtype='uint32')

        rowinds = np.array(([rowinds],) * colslength).transpose().flatten()
        colinds = np.array(([colinds],) * rowslength).flatten()
        weights = np.outer(rowweights, colweights).flatten()

        return np.sum(arrayin[rowinds, colinds] * areaweights * weights)

    curarray = np.array(curarray, dtype='float64')
    rowinds = np.array([range(0, 180), ] * 360).transpose().flatten()
    colinds = np.array([range(0, 360), ] * 180).flatten()
    newarraytest = map(lambda rows, cols: summingit(curarray, 2160, 4320, 180, 360, rows, cols), rowinds, colinds)
    newarraytest = np.reshape(newarraytest, (180, 360))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    states = os.path.join('IrrigationGIS', 'boundaries', 'states', 'western_states_11_wgs.shp')
    results = os.path.join('IrrigationGIS', 'results_figures', 'current_irr_area.tif')
    plot_map(states, results)
# ========================= EOF ====================================================================

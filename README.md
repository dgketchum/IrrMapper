Note: This repository are computer vision approaches to irrigated lands mapping. To see Earth Engine-based code referred to in our [Remote Sensing](https://www.mdpi.com/2072-4292/12/14/2328) article, go to [EEMapper](https://github.com/dgketchum/EEMapper)

# IrrMapper
Automate Mapping of Irrigated Lands

Installation Instructions:

This package is a little difficult to create a working python interpreter for.

First, get [Anaconda](anaconda.org) and [git](https://git-scm.com/), these tools
are important here.

Next, create your environment.

``` conda create -n irri python=3.6```

Then get the latest gdal:

``` conda install -c conda-forge gdal=2.2.3```

Then the latest master branch of rasterio:

```pip install git+https://github.com/mapbox/rasterio.git```

Install Metio:

```pip install git+https://github.com/tcolligan4/Metio.git```

Install SatelliteImage:

```pip install git+https://github.com/dgketchum/satellite_image.git```


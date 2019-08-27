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

# usage
## 1 
Edit the file ```runspec.py``` and fill out the two methods ```assign_shapefile_class_code``` and ```assign_shapefile_year```. These functions take as input a path to a shapefile and return integers corresponding to the class code of the vector data in the shapefile and the year the data was recorded, respectively. This means shapefiles should be split up according to class code and year before the next step.

Also in ``runpsec.py``, select the bands you want by editing ``landsat_rasters()``, ``static_rasters()``, and ``climate_rasters()``. ``mask_rasters()`` specifies water and cloud masks. 
## 2
use split_shapefile.py to split the training data shapefiles into WRS2 descending path/rows.
```python split_shapefile.py --shapefile-dir /home/thomas/training_data/ --output-dir /home/thomas/split_training_data/```
Default extension is .shp for the input shapefiles.
## 3
run extract_training_data.py to extract training data. This relies on the methods ``assign_shapefile_class_code`` and ``assign_shapefile_year`` that reside in ``runspec.py``. It also downloads all image data to image-dir. Right now, it downloads all 11 Landsat bands for 3 scenes from may-october. I need to figure out how to change this.
```python extract_training_data.py --shapefile-dir /home/thomas/split_training_data --image-dir /home/thomas/landsat/ --training-dir /home/thomas/irrmapper/data/train/ --n-classes 5```
Before running this, check ```_one_hot_from_labels_mc()``` in ```extract_training_data.py```. This applies a border class to shapefile data of class code 0 for reasons related to mapping irrigation. If this is not what you want, comment out the conditional in this function.  

## 4 
train a model with train_model.py.

# TODO: 
Make training a model easier (i.e. don't require a separate weights matrix and stop computing softmax within the network)
Make the images downloaded pull from runspec.py, not automatically download all possible bands
Implement IoU (multiclass dice loss).
add binary classification possibililty
add raster training data extraction
add paths_map_single_scene flag to extract_training_data









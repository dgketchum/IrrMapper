import glob
import os
from collections import defaultdict
import fiona
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros
import re
from rasterio import float32, open as rasopen
from compose_array_single_shapefile import PixelTrainingArraySingleShapefile
from prepare_images import ImageStack
import sat_image

def create_master_raster(image_stack):
     
    paths_map = image_stack.paths_map
    first = True
    stack = None

    for i, feat in enumerate(paths_map.keys()): # ordered dict ensures accuracy here.

        feature_raster = paths_map[feat] # maps bands to their location in filesystem.

        with rasopen(feature_raster, mode='r') as src:
            arr = src.read()
            raster_geo = src.meta.copy()

        if first:
            first_geo = deepcopy(raster_geo)
            empty = zeros((len(paths_map.keys()), arr.shape[1], arr.shape[2]), float32)
            stack = empty
            stack[i, :, :] = arr
            first = False
        else:
            try:
                stack[i, :, :] = arr
            except ValueError: 
                import pprint
                pprint.pprint(first_geo)
                # error was thrown here b/c source raster didn't have crs
                arr = sat_image.warped_vrt.warp_single_image(feature_raster, first_geo)
                stack[i, :, :] = arr

    # print(stack.shape)
    first_geo.update(count=len(paths_map.keys()))

    with rasopen("master_raster.tif", mode='w', **first_geo) as dst:
        dst.write(stack) 

    return "master_raster.tif" # TODO: replace with a sensible name.

def get_path_row(lat, lon):
    """
    :param lat: Latitude float
    :param lon: Longitude float
            'convert_pr_to_ll' [path, row to coordinates]
    :return: lat, lon tuple or path, row tuple
    """
    conversion_type = 'convert_ll_to_pr'
    base = 'https://landsat.usgs.gov/landsat/lat_long_converter/tools_latlong.php'
    unk_number = 1508518830987

    full_url = '{}?rs={}&rsargs[]={}&rsargs[]={}&rsargs[]=1&rsrnd={}'.format(base,
                                                                             conversion_type,
                                                                             lat, lon,
                                                                             unk_number)
    r = get(full_url)
    tree = html.fromstring(r.text)

    # remember to view source html to build xpath
    # i.e. inspect element > network > find GET with relevant PARAMS
    # > go to GET URL > view source HTML
    p_string = tree.xpath('//table/tr[1]/td[2]/text()')
    path = int(re.search(r'\d+', p_string[0]).group())

    r_string = tree.xpath('//table/tr[1]/td[4]/text()')
    row = int(re.search(r'\d+', r_string[0]).group())

    return path, row

def get_shapefile_lat_lon(shapefile):
    with fiona.open(shapefile, "r") as src:
        minx, miny, maxx, maxy = src.bounds
        latc = (maxy + miny) / 2
        lonc = (maxx + minx) / 2

    return latc, lonc 

def get_shapefile_path_rows(shapefile):
    '''Since ImageStack can deal with lat/lon when
       downloading from the Internet, I need to get
       a central location of the shapefile. However,
       since shapefiles may extend over multiple p/rs,
       I need to figure out all the path/rows.
       These methods work; however, they don't account
       for the overlapping nature of the wrs2 tiles.
       This means that many, many images may be downloaded.
       '''

    dct = defaultdict(list) 
    from shapely.geometry import shape
    with fiona.open(shapefile, "r") as src:
        for feat in src:
            pt = shape(feat['geometry']).centroid
            pt = pt.coords[0]
            p, r = get_path_row(pt[1], pt[0])
            pr = str(p) + "_" + str(r)
            dct[pr].append(feat)
    return dct 

def split_shapefile(path, data_directory, base_shapefile, pr_dict):
    ''' 
    Multiple p/rs can be contained in one
    shapefile. To ease the extraction of
    training data, we need an automated way 
    to look through all shapefile attributes and assign 
    them a path/row, then save all the polygons in different
    files (if they're not in the same p/r) 
    '''

    prefix = os.path.splitext(base_shapefile)[0]

    with fiona.open(os.path.join(path, base_shapefile), "r") as src:
        meta = deepcopy(src.meta)

    for pr in pr_dict.keys():
        out = prefix + "_" + pr + ".shp"
        print(out)
        with fiona.open(os.path.join(data_directory, out), 'w', **meta) as dst:
            for poly in pr_dict[pr]:
                dst.write(poly)
    return 

def download_images(project_directory, path, row, year):

    image_stack = ImageStack(satellite=8, path=path, row=row, root=project_directory,
            max_cloud_pct=70, n_landsat=3, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack

    
def create_training_data(shapefile, shapefile_directory, image_directory, class_code,
        kernel_size, instances, training_directory, year):

    latc, lonc = get_shapefile_lat_lon(os.path.join(shapefile_directory, shapefile))
    p, r = get_path_row(latc, lonc)
    suff = str(p) + '_' + str(r) + "_" + str(year)
    landsat_dir = os.path.join(image_directory, suff) 

    if not os.path.isdir(landsat_dir):
        os.mkdir(landsat_dir)
        ims = download_images(landsat_dir, p, r, year)
    else:
        print("Images may have been downloaded for {}_{}_{}".format(p, r, year))
        print("Check to make sure they're all there.")
        ims = download_images(landsat_dir, p, r, year)

    shp = os.path.join(shapefile_directory, shapefile)
    pta = PixelTrainingArraySingleShapefile(root=landsat_dir, shapefile_path=shp,
            paths_map=ims.paths_map, masks=ims.masks, class_code=class_code, 
            instances=instances, kernel_size=kernel_size, path=p, row=r)

    pta.extract_sample()
    prefix = os.path.splitext(shapefile)[0]
    prefix += "_data_kernel_{}".format(kernel_size)

    pta.to_pickle(pta.data, os.path.join(training_directory, prefix))

    print("Data saved to ".format(training_directory))

    
if __name__ == "__main__":

    # Workflow:
    # split all shapefiles up into separate path/rows.
    # Then for each (split) shapefile, download the
    # requisite image data (landsat -> et etc).
    # Then, for each split shapefile, 
    # download training data and save in the training data directory. 
    # TODO: 1. Make a program to train the model 
    # 2. Figure out how to make the model evaluate on every path/row
    # that's present in the training data. 
    # The code broke on an SSEBop run, something 
    # that I have no control over. At least I think I don't.   
    # shpfile = '/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_other_3728.shp'
    # path = '/home/thomas/IrrigationGIS/western_states_irrgis/MT/'
    # pr_dct = get_shapefile_path_rows(shpfile)

    train_dir = '../training_data'
    data_directory = '../shapefile_data/'
    image_directory = '../image_data'

    for f in glob.glob(data_directory + "*.shp"):
        if "sample_points" not in f:
            shp = os.path.basename(f) 
            print(shp)
            create_training_data(shp, data_directory, image_directory, 3, 17, 10, 
                train_dir, 2013) 

    #  
    # data_directory = '../shapefile_data/'
    # shpfile = '/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_other_3728.shp'
    # pr_dct = get_shapefile_path_rows(shpfile)
    # image_directory = '../image_data'
    # split_shapefile(path, data_directory, 'MT_other_3728.shp', pr_dct)

#   #   now create imagestacks from each shapefile in the split shapefile directory. 

    # # shapefile_year_map = empty

    # year = 2013 # I think?
    # for f in glob.glob(data_directory + '*.shp'):
    #     latc, lonc = get_shapefile_lat_lon(f)
    #     p, r = get_path_row(latc, lonc)
    #     suff = str(p) + '_' + str(r) + "_" + str(year)
    #     landsat_dir = os.path.join(image_directory, suff) 
    #     if not os.path.isdir(landsat_dir):
    #         os.mkdir(landsat_dir)
    #         ims = download_images(landsat_dir, p, r, 2013)
    #     else:
    #         print("Images may have been downloaded for {}_{}_{}".format(p, r, year))
    # print(i)
# project_directory, shapefile, year)


# download_images
    
    







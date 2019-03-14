import glob
import os
import geopandas as gpd
import json
from fiona import open as fopen
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros, asarray, array, reshape, nan, sqrt, std
from shapely.geometry import shape
from collections import defaultdict
from rasterio import float32, open as rasopen
from rasterio.mask import mask
from prepare_images import ImageStack
from sklearn.neighbors import KDTree
from sat_image.warped_vrt import warp_single_image


def get_features(gdf):
    tmp = json.loads(gdf.to_json())
    features = [feature['geometry'] for feature in tmp['features']]
    return features


def generate_class_mask(shapefile, master_raster, no_data=-1):
    ''' Generates a mask with class_val everywhere 
    shapefile data is present and a no_data value everywhere else.
    no_data is -1 in this case, as it is never a valid class label.
    Switching coordinate reference systems is important here, or 
    else the masking won't work.
    '''
    shp = gpd.read_file(shapefile)
    with rasopen(master_raster, 'r') as src:
        shp = shp.to_crs(src.crs)
        features = get_features(shp)
        out_image, out_transform = mask(src, shapes=features, nodata=no_data)
    return out_image


def create_master_raster(paths_map, path, row, year, raster_directory, mean_map, stddev_map):
    """ Creates a master raster with depth given by the organization of the
    paths_map. Paths map is a dictionary of lists, with keys the band names 
    (B1, B2...) and values the paths of the images in the filesystem 
    corresponding to that band. """
    fname = "master_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)
    mask_fname = "class_mask_{}_{}_{}.tif".format(path, row, year)
    mask_path = os.path.join(raster_directory, mask_fname)
    if os.path.isfile(pth):
        print("Master raster already created for {}_{}_{}.".format(path, row, year))
        if os.path.isfile(mask_path):
            print('Class mask template already created for {}_{}_{}'.format(path, row, year))
            return pth
        else:
            print("Creating class mask template.")
            with rasopen(pth, 'r') as src:
                meta = src.meta.copy()
                h = meta['height']
                w = meta['width']

            meta.update(count=1, dtype=float32)

            with rasopen(mask_path, 'w', **meta) as msk:
                out = zeros((h, w)).astype(float32)
                msk.write(out, 1)
            return pth
        
    first = True
    stack = None
    num_rasters = 0
    for key in paths_map:
        num_rasters += len(paths_map[key])

    j = 0
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's essentially sorting by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            band_mean = None
            for band in mean_map:
                if feature_raster.endswith(band):
                    band_mean = mean_map[band]
                    band_std = stddev_map[band]

            if band_mean is None:
                print("Band mean not found in mean_mapping for {}".format(feature_raster))
                return

            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
                raster_geo = src.meta.copy()

            #arr = (arr - band_mean) / band_std
            arr = (arr - arr.mean()) / std(arr)

            if first:
                first_geo = raster_geo.copy()
                empty = zeros((num_rasters, arr.shape[1], arr.shape[2]), float32)
                stack = empty
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1
                except ValueError: 
                    # error can be thrown here if source raster doesn't have crs
                    # OR ! Because rasterio version.
                    # However, deepcopy becomes an issue with the latest
                    # version of rasterio.
                    arr = warp_single_image(feature_raster, first_geo)
                    stack[j, :, :] = arr
                    j += 1

    msk_out = zeros((1, stack.shape[1], stack.shape[2]))
    first_geo.update(count=1, dtype=msk_out.dtype)
    with rasopen(mask_path, mode='w', **first_geo) as msk:
        msk.write(msk_out)

    first_geo.update(count=num_rasters, dtype=stack.dtype)

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack)

    print("Master raster saved to {}.".format(pth))

    return pth


def get_shapefile_lat_lon(shapefile):
    ''' Center of shapefile'''
    with fopen(shapefile, "r") as src:
        minx, miny, maxx, maxy = src.bounds
        latc = (maxy + miny) / 2
        lonc = (maxx + minx) / 2

    return latc, lonc 


def normalize_and_save_image(fname):
    norm = True
    with rasopen(fname, 'r') as rsrc:
        if "normalized" in rsrc.tags():
            return
        else:    
            rass_arr = rsrc.read()
            rass_arr = rass_arr.astype(float32)
            profile = rsrc.profile.copy()
            profile.update(dtype=float32)
            rass_arr = rass_arr.reshape(rass_arr.shape[1], rass_arr.shape[2])
            scaler = StandardScaler() # z-normalization
            scaler.fit(rass_arr)
            rass_arr = scaler.transform(rass_arr)
    with rasopen(fname, 'w', **profile) as dst:
        dst.write(rass_arr, 1)
        print("Normalized", fname)
        dst.update_tags(normalized=True)


def raster_sum(raster):
    with rasopen(raster, 'r') as src:
        arr_masked = src.read(1, masked=True) # get rid of nodata values
    s = arr_masked.sum()
    count = arr_masked.count()
    return s, count


def raster_squared_sum(raster, mean):
    with rasopen(raster, 'r') as src:
        arr_masked = src.read(1, masked=True) # get rid of nodata values
    squared_diff = (arr_masked - mean)**2
    s = squared_diff.sum()
    count = squared_diff.count()
    return s, count


def bandwise_stddev(paths_list, band_name, band_mean):
    ''' Calculate the stddev of the pixel
    values in a given band through time.'''
    n_pixels = 0
    pixel_value_squared_sum = 0
    for filepath in paths_list:
        p_sum, num_pix = raster_squared_sum(filepath, band_mean)
        pixel_value_squared_sum += p_sum
        n_pixels += num_pix
    if n_pixels == 0:
        print("0 non masked pixels.")
        return 1
    return (sqrt(pixel_value_squared_sum / n_pixels), band_name)


def bandwise_mean(paths_list, band_name):
    ''' Need to center the data to have 
    a zero mean. This means iterating over all images, 
    and taking the "band-wise" mean, then subtracting
    that mean from the band. This mean should
    also only be computed for the test set, but applied
    to the training set. ''' 
    n_pixels = 0
    pixel_value_sum = 0
    for filepath in paths_list:
        p_sum, num_pix = raster_sum(filepath)
        pixel_value_sum += p_sum
        n_pixels += num_pix
    if n_pixels == 0:
        print("0 non masked pixels.")
        return 1
    return (pixel_value_sum / n_pixels, band_name)


def download_images(project_directory, path, row, year, satellite=8, n_landsat=3):

    image_stack = ImageStack(satellite=satellite, path=path, row=row, root=project_directory,
            max_cloud_pct=70, n_landsat=n_landsat, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack

def construct_kdtree(wrs2):
    centroids = []
    path_rows = [] # a mapping
    features = []
    for feature in wrs2:
        tile = shape(feature['geometry'])
        centroid = tile.centroid.coords[0]
        centroids.append([centroid[0], centroid[1]])
        z = feature['properties']
        p = z['PATH']
        r = z['ROW']
        path_rows.append(str(p) + "_" + str(r))
        features.append(feature)

    tree = KDTree(asarray(centroids))
    return tree, asarray(path_rows), asarray(features)

def get_pr(poly, wrs2):
    ls = []
    for feature in wrs2:
        tile = shape(feature['geometry'])
        if poly.within(tile):
            z = feature['properties']
            p = z['PATH']
            r = z['ROW']
            ls.append(str(p) + "_" + str(r))
    return ls

def get_pr_subset(poly, tiles):
    ''' Use when you only want to iterate
    over a subset of wrs2 tiles.'''
    ls = []
    for feature in tiles:
        tile = shape(feature['geometry'])
        if poly.within(tile):
            z = feature['properties']
            p = z['PATH']
            r = z['ROW']
            ls.append(str(p) + "_" + str(r))
    return ls

def filter_shapefile(shapefile, out_directory): 
    """ Shapefiles may span multiple path/rows.
    For training, we want all of the data available.
    This function filters the polygons contained in
    the shapefile into separate files for each path/row
    contained in the shapefile. """
    path_row_map = defaultdict(list)
    wrs2 = fopen('../spatial_data/wrs2_descending_usa.shp', 'r')
    tree, path_rows, features = construct_kdtree(wrs2)
    wrs2.close()

    cent_arr = array([0, 0])
    with fopen(shapefile, "r") as src:
        meta = deepcopy(src.meta)
        for feat in src:
            poly = shape(feat['geometry'])
            centroid = poly.centroid.coords[0]
            cent_arr[0] = centroid[0]
            cent_arr[1] = centroid[1]
            centroid = cent_arr.reshape(1, -1)
            dist, ind = tree.query(centroid, k=10)
            tiles = features[ind[0]]
            prs = get_pr_subset(poly, tiles)
            for p in prs:
                path_row_map[p].append(feat)

    outfile = os.path.basename(shapefile)
    outfile = os.path.splitext(outfile)[0]

    for path_row in path_row_map:
        out = outfile + path_row + ".shp"
        with fopen(os.path.join(out_directory, out), 'w', **meta) as dst:
            print("Saving {}".format(out))
            for feat in path_row_map[path_row]:
                dst.write(feat)


def split_shapefile(base, base_shapefile, data_directory):
    """
    Shapefiles may deal with data over multiple path/rows.
    This is a method to get the minimum number of
    path/rows required to cover all features. 
    Data directory: where the split shapefiles will be saved.
    base: directory containing base_shapefile."""
    path_row = defaultdict(list) 
    id_mapping = {}
    # TODO: un hardcode this directory.
    wrs2 = fopen('../spatial_data/wrs2_descending_usa.shp', 'r')
    tree, path_rows, features = construct_kdtree(wrs2)
    wrs2.close()

    cent_arr = array([0, 0])
    with fopen(os.path.join(base, base_shapefile), "r") as src:
        meta = deepcopy(src.meta)
        for feat in src:
            idd = feat['id']
            id_mapping[idd] = feat
            poly = shape(feat['geometry'])
            centroid = poly.centroid.coords[0]
            cent_arr[0] = centroid[0]
            cent_arr[1] = centroid[1]
            centroid = cent_arr.reshape(1, -1)
            dist, ind = tree.query(centroid, k=10)
            tiles = features[ind[0]]
            prs = get_pr_subset(poly, tiles)
            for p in prs:
                path_row[p].append(idd)

    non_unique_ids = defaultdict(list)
    unique = defaultdict(list)
    for key in path_row:
        ls = path_row[key] # all features in a given path/row
        placeholder = ls.copy()
        for key1 in path_row:
            if key != key1:
                ls1 = path_row[key1]
                # find unique keys in ls
                placeholder = set(placeholder) - set(ls1) #all 
                # features present in placeholder that are not 
                # present in ls1; i.e. unique keys
        unique[key] = list(placeholder)
        if len(ls) != len(placeholder): 
            nu = set(ls) - set(placeholder) # all features present in ls that are not present in placeholder (non-unique)
            for idd in list(nu):
                non_unique_ids[idd].append(key)
   
    match_key = []
    for key in non_unique_ids: # unique ids 
        pr = None 
        hi = 0
        for pathrow in non_unique_ids[key]: # path/rows corresponding to non
            # unique features
            if len(unique[pathrow]) > hi:
                pr = pathrow 
                hi = len(unique[pathrow])

        if pr is not None:
            unique[pr].append(key)
        else:
            choice = non_unique_ids[key]
            choice.sort()
            choice = choice[0]
            unique[choice].append(key)

    prefix = os.path.splitext(base_shapefile)[0]
    for key in unique:
        if key is None:
            continue
        out = prefix + "_" + key + ".shp"
        if len(unique[key]):
            with fopen(os.path.join(data_directory, out), 'w', **meta) as dst:
                print("Saving split shapefile to: {}".format(os.path.join(data_directory, out)))
                for feat in unique[key]:
                    dst.write(id_mapping[feat])


def get_shapefile_path_row(shapefile):
    """This function assumes that the original
    shapefile has already been split, and relies on
    the naming convention to get the path and row.  """
    # strip extension
    # TODO: Find some way to update shapefile metadata
    shp = shapefile[-9:-4].split("_")
    return int(shp[0]), int(shp[1])


def shapefile_area(shapefile):
    summ = 0
    with fopen(shapefile, "r") as src:
        for feat in src:
            poly = shape(feat['geometry'])
            summ += poly.area
    return summ


def get_total_area(data_directory, filenames):
    ''' Gets the total area of the polygons
        in the files in filenames
        TODO: Get an equal-area projection'''

    tot = 0
    for f in glob.glob(data_directory + "*.shp"):
        if "sample" not in f:
            for f2 in filenames:
                if f2 in f:
                    tot += shapefile_area(f)
    return tot


def required_points(shapefile, total_area, total_instances):
    area = shapefile_area(shapefile)
    frac = area / total_area
    return int(total_instances * frac)
    

if __name__ == "__main__":
    pass

from shapely.geometry import shape
import glob
import os
from collections import defaultdict
import fiona
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros, asarray, array, reshape, nan
from rasterio import float32, open as rasopen
from rasterio.mask import mask
from prepare_images import ImageStack
from sklearn.neighbors import KDTree
from sat_image.warped_vrt import warp_single_image
import geopandas as gpd
import json

NO_DATA = -1

def get_features(gdf):
    tmp = json.loads(gdf.to_json())
    features = [feature['geometry'] for feature in tmp['features']]
    return features

def generate_class_mask(shapefile, master_raster):
    ''' Generates a mask with class_val everywhere 
    shapefile data is present and a no_data value everywhere else.
    no_data is -1 in this case, as it is never a valid class label.
    Switching coordinate reference systems is important here, or 
    else the masking won't work.
    '''
    shp = gpd.read_file(shapefile)
    with rasopen(master_raster, 'r') as src:
        shp = shp.to_crs(src.crs)
        arr = src.read()
        features = get_features(shp)
        out_image, out_transform = mask(src, shapes=features, nodata=NO_DATA)
    return out_image

def create_master_masked_raster(image_stack, path, row, year, raster_directory):
    masks = image_stack.masks
    if not masks:
        return None

    first = True
    stack = None

    for i, feat in enumerate(masks.keys()): # ordered dict ensures accuracy here.

        mask_raster = masks[feat] # maps bands to their location in filesystem.

        with rasopen(mask_raster, mode='r') as src:
            arr = src.read()
            raster_geo = src.meta.copy()

        if first:
            first_geo = deepcopy(raster_geo)
            empty = zeros((len(masks.keys()), arr.shape[1], arr.shape[2]), float32)
            stack = empty
            stack[i, :, :] = arr
            first = False
        else:
            try:
                stack[i, :, :] = arr
            except ValueError: 
                arr = warp_single_image(mask_raster, first_geo)
                stack[i, :, :] = arr

    first_geo.update(count=len(masks.keys()))

    fname = "master_mask_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack) 

    return pth


def create_master_raster(image_stack, path, row, year, raster_directory):
    fname = "master_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)
    mask_fname = "class_mask_{}_{}_{}.tif".format(path, row, year)
    mask_path = os.path.join(raster_directory, mask_fname)
    if os.path.isfile(pth):
        print("Master raster already created for {}_{}_{}.".format(path, row, year))
        if os.path.isfile(mask_path):
            print('Class mask template already created')
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
                # error can be thrown here if source raster doesn't have crs
                arr = warp_single_image(feature_raster, first_geo)
                stack[i, :, :] = arr

    first_geo.update(count=1)
    msk_out = zeros((stack.shape[1], stack.shape[0]))
    with rasopen(mask_path, mode='w', **first_geo) as msk:
        msk.write(msk_out)

    first_geo.update(count=len(paths_map.keys()))

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack)

    return pth

def get_shapefile_lat_lon(shapefile):
    ''' Center of shapefile'''
    with fiona.open(shapefile, "r") as src:
        minx, miny, maxx, maxy = src.bounds
        latc = (maxy + miny) / 2
        lonc = (maxx + minx) / 2

    return latc, lonc 

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
    wrs2 = fiona.open('../spatial_data/wrs2_descending_usa.shp', 'r')
    tree, path_rows, features = construct_kdtree(wrs2)
    wrs2.close()

    cent_arr = array([0, 0])
    with fiona.open(os.path.join(base, base_shapefile), "r") as src:
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
            with fiona.open(os.path.join(data_directory, out), 'w', **meta) as dst:
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
    
if __name__ == "__main__":
    pass

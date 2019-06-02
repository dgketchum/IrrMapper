import geopandas as gpd
import os
from json import loads
from numpy import zeros, asarray, array, reshape, nan, sqrt, std
from copy import deepcopy
from fiona import open as fopen
from rasterio.mask import mask
from rasterio import open as rasopen
from shapely.geometry import shape, mapping
from sklearn.neighbors import KDTree
from collections import defaultdict


def get_features(gdf):
    tmp = loads(gdf.to_json())
    features = [feature['geometry'] for feature in tmp['features']]
    return features


def mask_raster_to_shapefile(shapefile, raster, return_binary=True):
    ''' Generates a mask with 1 everywhere 
    shapefile data is present and a no_data value everywhere else.
    no_data is -1 in this case, as it is never a valid class label.
    Switching coordinate reference systems is important here, or 
    else the masking won't work.
    '''
    shp = gpd.read_file(shapefile)
    shp = shp[shp.geometry.notnull()]
    with rasopen(raster, 'r') as src:
        shp = shp.to_crs(src.crs)
        features = get_features(shp)
        arr = src.read()
        out_image, out_transform = mask(src, shapes=features)
        if return_binary:
            out_image[out_image != 0] = 1 
        meta = src.meta
    return out_image, meta


def generate_class_mask(shapefile, master_raster, nodata=-1):
    ''' Generates a mask with 1 everywhere 
    shapefile data is present and a no_data value everywhere else.
    no_data is -1 in this case, as it is never a valid class label.
    Switching coordinate reference systems is important here, or 
    else the masking won't work.
    '''
    shp = gpd.read_file(shapefile)
    shp = shp[shp.geometry.notnull()]
    with rasopen(master_raster, 'r') as src:
        shp = shp.to_crs(src.crs)
        features = get_features(shp)
        out_image, out_transform = mask(src, shapes=features, nodata=-1)
        out_image[out_image != -1] = 1 
        out_image[out_image == -1] = 0
        meta = src.meta
    return out_image, meta


def get_shapefile_lat_lon(shapefile):
    ''' Center of shapefile'''
    with fopen(shapefile, "r") as src:
        minx, miny, maxx, maxy = src.bounds
        latc = (maxy + miny) / 2
        lonc = (maxx + minx) / 2

    return latc, lonc 


def _construct_kdtree(wrs2):
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


def filter_shapefile_overlapping(shapefile, save=False, out_directory=None): 
    """ Shapefiles may span multiple path/rows/years.
    For training, we want all of the data available.
    This function filters the polygons contained in
    the shapefile into separate files for each path/row/year
    contained in the shapefile. """
    path_row_map = defaultdict(list)
    wrs2 = fopen('../spatial_data/wrs2_descending_usa.shp', 'r')
    tree, path_rows, features = _construct_kdtree(wrs2)
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
            prs = get_pr_subset(poly, tiles) # gets the matching path/rows

            for p in prs:
                path_row_map[p].append(feat)

    if not save:
        return path_row_map

    outfile = os.path.basename(shapefile)
    outfile = os.path.splitext(outfile)[0]

    for path_row in path_row_map:
        out = outfile + path_row + ".shp"
        with fopen(os.path.join(out_directory, out), 'w', **meta) as dst:
            print("Saving {}".format(out))
            for feat in path_row_map[path_row]:
                dst.write(feat)


def filter_shapefile_non_overlapping(base, base_shapefile, data_directory):
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
    tree, path_rows, features = _construct_kdtree(wrs2)
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


def buffer_shapefile(shp):
    buf = -0.00050
    with fopen(shp, 'r') as polys:
        out = []
        meta = polys.meta
        with fopen(shp, 'w', **meta) as dst:
            for feat in polys:
                feat['geometry'] = mapping(shape(feat['geometry']).buffer(buf))
                dst.write(feat)

if __name__ == '__main__':
    from glob import glob
    out_dir = '/home/thomas/IrrigationGIS/UT_CO_MT_WY_split/'
    for f in glob('/home/thomas/IrrigationGIS/UT_CO_MT_WY/' + "*.shp"):
        if 'unirrigated' in f:
            filter_shapefile(f, out_dir)

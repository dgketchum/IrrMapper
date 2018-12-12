import fiona
from rasterio import float32, open as rasopen
from prepare_images import ImageStack
from sat_image.warped_vrt import warp_single_image

def get_shapefile_center(shapefile):
    '''Since ImageStack can deal with lat/lon when
       downloading from the Internet, I need to get
       a central location of the shapefile'''

    with fiona.open(shapefile) as src:
        bounds = src.bounds # returns minx, miny. maxx, maxy
        (minx, miny, maxx, maxy) = bounds
        latc = (maxy + miny)/2
        lonc = (minx + maxx)/2
    
    return latc, lonc 

def download_images(project_directory, shapefile, year):

    lat, lon = get_shapefile_center(shapefile)

    image_stack = ImageStack(satellite=8, lat=lat, lon=lon, root=project_directory,
            max_cloud_pct=70, n_landsat=3, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask. Evaluating excludes a cloud mask
    # need to save paths_map and masks 
    return image_stack  


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
                geo = deepcopy(raster_geo)
                empty = zeros((len(paths_map.keys()), arr.shape[1], arr.shape[2]), float32)
                stack = empty
                stack[i, :, :] = arr
                first = False
            else:
                try:
                    stack[i, :, :] = arr
                except ValueError:
                    arr = warp_single_image(self.feature_ras, first_geo)
                    stack[i, :, :] = arr

    with rasopen("master_raster.tif", mode='w', **geo) as dst:
        dst.write(stack)

    return "master_raster.tif" # TODO: replace with a sensible name.
    
def create_training_data(project_directory, shapefile, year, lat, lon):
    # why don't we create a master masked raster?
    # then use the same image for evaluation and 
    # training. 
    image_stack = download_images(project_directory, shapefile, year)
    
if __name__ == "__main__":

    shpfile = '/home/thomas/IrrigationGIS/western_states_irrgis/MT/MT_3728.shp'

    iss = download_images("../model_data/allstates_3/MT", shpfile, 2017)
    string = create_master_raster(iss)


    
    







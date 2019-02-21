import numpy as np
import os
from glob import glob
from data_utils import generate_class_mask, get_shapefile_path_row
from rasterio import open as rasopen

NO_DATA = -1
MAX_POOLS = 3
CHUNK_SIZE = 1248 # some value that is evenly divisible by 2^3.
NUM_CLASSES = 2

def random_sample(class_mask, n_instances, box_size, class_code=1):
    out = np.where(class_mask != NO_DATA)
    class_mask = class_mask.copy()
    try:
        out_x = out[1]
        out_y = out[2] 
    except IndexError as e:
        out_x = out[0]
        out_y = out[1] 

    indices = np.random.choice(len(out_x), size=n_instances, replace=False)
    out_x = out_x[indices]
    out_y = out_y[indices] 

    try:
        class_mask[:, :, :] = NO_DATA
        if box_size == 0:
            class_mask[0, out_x, out_y] = class_code
        else:
            ofs = box_size // 2
            for x, y in zip(out_x, out_y):
                class_mask[0, x-ofs:x+ofs+1, y-ofs:y+ofs+1] = class_code

    except IndexError as e:
        class_mask[:, :] = NO_DATA
        if box_size == 0:
            class_mask[out_x, out_y] = class_code
        else:
            ofs = box_size // 2
            for x, y in zip(out_x, out_y):
                class_mask[x-ofs:x+ofs, y-ofs:y+ofs] = class_code

    return class_mask

def load_raster(master_raster):
    with rasopen(master_raster, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta

def generate_binary_train(shapefile_directory, image_directory, box_size, target):

    while True:
        for f in glob(os.path.join(shapefile_directory, "*.shp")):
            if target in f:
                all_matches = all_matching_shapefiles(f, shapefile_directory)
                p, r = get_shapefile_path_row(f)
                suffix = '{}_{}_{}.tif'.format(p, r, year)
                master_raster = os.path.join(image_directory, train_raster + suffix)
                mask_file = os.path.join(image_directory, mask_raster + suffix)
                if not os.path.isfile(master_raster):
                    print("Master raster not created for {}".format(suffix))
                    # TODO: More extensive error handling.
                else:
                    target_mask = generate_class_mask(f, mask_file)
                    class_mask = np.ones((NUM_CLASSES, target_mask.shape[1], target_mask.shape[2]))*NO_DATA
                    class_mask[1, :, :] = target_mask
                    required_instances = len(np.where(target_mask != NO_DATA)[0]) // (box_size*len(all_matches))
                    masks = []
                    for match in all_matches:
                        msk = generate_class_mask(match, mask_file)
                        samp = random_sample(msk, required_instances, box_size)
                        masks.append(samp)
                    for i, s in enumerate(masks):
                        class_mask[0, :, :][s[0, :, :] != NO_DATA] = 1
                    # May need to do some preprocessing.    
                    master = load_raster(master_raster)
                    for i in range(0, master.shape[1], CHUNK_SIZE):
                        for j in range(0, master.shape[2], CHUNK_SIZE):
                            sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            sub_master, sub_mask = preprocess_data(sub_master, sub_mask)
                            if np.all(sub_mask == NO_DATA):
                                continue
                            else:
                                yield sub_master, sub_mask


def all_matching_shapefiles(to_match, shapefile_directory):
    out = []
    pr = get_shapefile_path_row(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and to_match not in f:
            out.append(f)
    return out

def generate_balanced_data(shapefile_directory, image_directory, box_size, target, year=2013):
    train_raster = 'master_raster_' # templates
    mask_raster = 'class_mask_'
    ''' This is pretty much for binary classification.'''
    while True:
        for f in glob(os.path.join(shapefile_directory, "*.shp")):
            if target in f:
                all_matches = all_matching_shapefiles(f, shapefile_directory)
                p, r = get_shapefile_path_row(f)
                suffix = '{}_{}_{}.tif'.format(p, r, year)
                master_raster = os.path.join(image_directory, train_raster + suffix)
                mask_file = os.path.join(image_directory, mask_raster + suffix)
                if not os.path.isfile(master_raster):
                    print("Master raster not created for {}".format(suffix))
                    # TODO: More extensive error handling.
                else:
                    target_mask = generate_class_mask(f, mask_file)
                    class_mask = np.ones((NUM_CLASSES, target_mask.shape[1], target_mask.shape[2]))*NO_DATA
                    class_mask[1, :, :] = target_mask
                    required_instances = len(np.where(target_mask != NO_DATA)[0]) // (box_size*len(all_matches))
                    masks = []
                    for match in all_matches:
                        msk = generate_class_mask(match, mask_file)
                        #samp = random_sample(msk, required_instances, box_size)
                        #masks.append(samp)
                        masks.append(msk)

                    for i, s in enumerate(masks):
                        class_mask[0, :, :][s[0, :, :] != NO_DATA] = 1

                    master, meta = load_raster(master_raster)
                    for i in range(0, master.shape[1], CHUNK_SIZE):
                        for j in range(0, master.shape[2], CHUNK_SIZE):
                            sub_master = master[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            sub_mask = class_mask[:, i:i+CHUNK_SIZE, j:j+CHUNK_SIZE]
                            if np.all(sub_mask == NO_DATA):
                                continue
                            else:
                                n_negative = len(np.where(sub_mask[0, :, :] != NO_DATA)[1]) 
                                positive = np.where(target_mask[:, :] != NO_DATA)
                                sorted_x = sorted(positive[1])
                                sorted_y = sorted(positive[2])
                                l = len(sorted_x) // 2
                                center_x = sorted_x[l]
                                center_y = sorted_y[l]
                                ofs = CHUNK_SIZE // 2
                                sub_positive = target_mask[:, center_x - ofs: center_x + ofs, center_y - ofs: center_y + ofs]
                                sub_master_positive = master[:, center_x - ofs: center_x + ofs, center_y - ofs: center_y + ofs]
                                required_instances = min(len(np.where(sub_positive[0, :, :] != NO_DATA)[1]), n_negative)
                                sub_negative = random_sample(sub_mask[0, :, :], required_instances,
                                        box_size=0, class_code=1)
                                sub_master_negative = sub_master
                                sub_positive = random_sample(sub_positive[0, :, :], required_instances,
                                       box_size=0, class_code=1)
                                one_hot_pos = np.ones((2, sub_positive.shape[0], sub_positive.shape[1]))*NO_DATA
                                one_hot_neg = np.ones((2, sub_negative.shape[0], sub_negative.shape[1]))*NO_DATA
                                one_hot_pos[1, :, :] = sub_positive
                                one_hot_neg[0, :, :] = sub_negative
                                sub_mas_pos, class_mask_pos = preprocess_data(sub_master_positive,
                                        one_hot_pos)
                                sub_mas_neg, class_mask_neg = preprocess_data(sub_master_negative,
                                        one_hot_neg)
                                ims = [sub_mas_pos, sub_mas_neg]
                                class_masks = [class_mask_pos, class_mask_neg]
                                for ii, jj in zip(ims, class_masks):
                                    yield ii, jj


def rotation(image, angle):
    return transform.rotate(image, angle, mode='constant', cval=NO_DATA)

def random_noise(image):
    return util.random_noise(image)

def h_flip(image):
    return image[:, ::-1]

def augment_data(image, class_mask):
    '''Randomly augments an image.'''
    if np.random.randint(2):
        deg = np.random.uniform(-25, 25)
        image = rotation(image, deg)
        class_mask = rotation(class_mask, deg)
    if np.random.randint(2):
        image = random_noise(image)
    if np.random.randint(2):
        image = h_flip(image)
        class_mask = h_flip(class_mask)
    if np.random.randint(2):
        image = np.flipud(image)
        class_mask = np.flipud(class_mask)
    return image, class_mask


def preprocess_data(master, mask, return_cuts=False):
    shp = master.shape
    rows = shp[1]; cols = shp[2]
    cut_rows = rows % (2**MAX_POOLS) 
    cut_cols = cols % (2**MAX_POOLS)
    out_m = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))

    if cut_cols != 0 and cut_rows != 0:
        out_m[0, :, :, :] = master[:, :-cut_rows, :-cut_cols]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :-cut_rows, :-cut_cols]
    elif cut_cols == 0 and cut_rows != 0:
        out_m[0, :, :, :] = master[:, :-cut_rows, :]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :-cut_rows, :]
    elif cut_cols != 0 and cut_rows == 0:
        out_m[0, :, :, :] = master[:, :, :-cut_cols]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :, :-cut_cols]
    else:
        out_m[0, :, :, :] = master[:, :, :]
        shp = mask.shape
        out_mask = np.zeros((1, shp[0], shp[1] - cut_rows, shp[2] - cut_cols))
        out_mask[0, :, :, :] = mask[:, :, :]

    out_m = np.swapaxes(out_m, 1, 3)
    out_mask = np.swapaxes(out_mask, 1, 3)
    if return_cuts:
        return out_m, out_mask, cut_rows, cut_cols

    return out_m, out_mask

if __name__ == '__main__':
    pass

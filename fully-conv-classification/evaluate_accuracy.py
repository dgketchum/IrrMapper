import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint
from sys import argv
from data_utils import load_raster
from data_generators import assign_class_code, concatenate_fmasks
from sklearn.metrics import confusion_matrix, jaccard_similarity_score
from shapefile_utils import generate_class_mask, get_shapefile_path_row


def evaluate_accuracy(argmaxed_raster, shapefile_test_dir, master_raster_dir, target_dict,
        show=False):
    shp_dict = {}
    # TODO: A weighted accuracy metric might be better.
    pr = None
    for f in glob(shapefile_test_dir + "*.shp"):
        pr = get_shapefile_path_row(f)
        cc = assign_class_code(target_dict, f)
        shp_dict[cc] = f

    class_mask_template = os.path.join(master_raster_dir, "class_mask_{}_{}_2013.tif".format(pr[0], pr[1]))
    first = True
    out = None
    nodata = -1
    for class_code in sorted(shp_dict.keys()):
        mask, mask_meta = generate_class_mask(shp_dict[class_code], class_mask_template,
                nodata)
        if first:
            out = np.ones((mask.shape[1], mask.shape[2], len(shp_dict)))*-1
            first = False
        out[:, :, class_code][mask[0] != nodata] = 1

    image_dir = '/home/thomas/share/image_data/test/{}_{}_2013'.format(pr[0], pr[1])
    mask = np.zeros_like(mask)
    fmask = concatenate_fmasks(image_dir, mask, mask_meta)
    for i in range(out.shape[2]):
        out[:, :, i][fmask[0] != 0] = -1

    bool_mask = np.not_equal(np.sum(out, axis=2), -4)
    y_pred, _ = load_raster(argmaxed_raster)
    if 'argmax' not in argmaxed_raster:
        y_pred = np.argmax(y_pred, axis=0)
    y_true = np.argmax(out, axis=2)

    for i in range(5):
        y_pred_irr = y_pred[y_true == i]
        print("Class {} acc: {}".format(i, np.sum(np.not_equal(y_pred_irr, i)) / y_pred_irr.size))

    y_pred_masked = y_pred[bool_mask]
    y_true_masked = y_true[bool_mask]
    print("Confusion mat for {} (all classes):".format(argmaxed_raster))
    cmat = confusion_matrix(y_true_masked, y_pred_masked)
    pprint(cmat)
    final = np.mean(np.equal(y_pred_masked, y_true_masked))
    print("pixel wise acc {}".format(final))
    print("Class precision:")
    print(np.diag(cmat) / np.sum(cmat, axis=0))
    print("Class recall:")
    print(np.diag(cmat) / np.sum(cmat, axis=1))
    if show:
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(y_pred[0])
        ax[1].imshow(y_true)
        ax[2].imshow(bool_mask)
        plt.suptitle('F: {} | acc: {}'.format(argmaxed_raster, final))
        plt.show()
    return final

if __name__ == '__main__':

    irr1 = 'Huntley'
    irr2 = 'Sun_River'
    fallow = 'Fallow'
    forest = 'Forrest'
    other = 'other'
    target_dict = {irr2:0, irr1:0, fallow:1, forest:2, other:3}
    shapefile_test_dir = 'shapefile_data/test/'
    master_raster_dir = '/home/thomas/share/master_rasters/test/'
    if len(argv) > 1:
        argmaxed_raster = argv[1]
        evaluate_accuracy(argmaxed_raster, shapefile_test_dir, master_raster_dir, target_dict)
    else:
        rsa = [f for f in glob('compare_model_outputs/during-the-day/' + '*.tif')]
        accs = {}
        for argmaxed_raster in rsa:
            print("-------------------------")
            print(argmaxed_raster)
            acc = evaluate_accuracy(argmaxed_raster, shapefile_test_dir, master_raster_dir, target_dict)
            accs[argmaxed_raster] = acc

        sort = sorted(acc.items(), key=lambda kv: kv[1])
        for key in sort:
            print("Raster: {} | acc: {}".format(key, accs[key]))

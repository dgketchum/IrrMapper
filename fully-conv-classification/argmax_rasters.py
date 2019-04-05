import numpy as np
from rasterio import open as rasopen
from rasterio import int32
from glob import glob
from os.path import basename, join
from sys import argv


im_path = 'compare_model_outputs/systematic/'
save_path = 'compare_model_outputs/argmax/'

def get_argmax(f, outfile):
    with rasopen(f, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()

    arg = np.argmax(arr, axis=0)
    arg = np.expand_dims(arg, axis=0)
    arg = arg.astype(np.int32)
    meta.update(count=1, dtype=int32)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arg)
    return None


def main(f):
    b = basename(f)
    suff = b[:-14]
    pref = b[-14:]
    outfile = join(save_path, suff + 'argmax_' + pref)
    print('Saving argmax raster to {}'.format(outfile))
    get_argmax(f, outfile)


if __name__ == '__main__':
    in_f = argv[1]
    main(in_f)

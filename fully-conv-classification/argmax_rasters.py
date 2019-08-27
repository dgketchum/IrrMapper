import numpy as np
from rasterio import open as rasopen
from rasterio import int32, float32
from glob import glob
from os.path import basename, join, dirname, splitext
import argparse

def compute_argmax(f, outfile):

    with rasopen(f, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()

    irr = arr[0]
    irr[irr < 0.3] = np.nan
    irr = np.expand_dims(irr, 0)
    arg = np.argmax(arr, axis=0)
    arg = np.expand_dims(arg, axis=0)
    arg = arg.astype(np.int32)
    meta.update(count=1, dtype=float32)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(irr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", 
            "--file",
            help='geoTIFF to perform argmax on',
            required=True)
    parser.add_argument('-o',
            '--outfile',
            help='optional filename for outfile')

    args = parser.parse_args()
    if not args.outfile:
        outfile = basename(args.file)
        outdir = dirname(args.file)
        outfile = splitext(outfile)[0] + '_argmax.tif'
        outfile = join(outdir, outfile)
        compute_argmax(args.file, outfile)






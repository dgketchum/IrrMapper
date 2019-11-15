import os
from argparse import ArgumentParser
from data_utils import clip_raster
from glob import glob

def _parse_path_row(f):
    bs = os.path.basename(f).split("_")
    return bs[0], bs[1]

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--raster', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--outfile', type=str)
    args = ap.parse_args()
    if args.outfile is None:
        outfile = args.raster
    path, row = _parse_path_row(args.raster)
    clip_raster(args.raster, int(path), int(row), outfile=outfile)

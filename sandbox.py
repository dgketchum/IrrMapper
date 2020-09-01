import os
import numpy as np
from pathlib import Path

path = Path(__file__).parents
PSE = os.path.join(path[0], 'data', 'pixel_sets', 'DATA')


def get_shapes(data):
    count_pixels = 0
    l = [os.path.join(data, x) for x in os.listdir(data)]
    for i, f in enumerate(l):
        a = np.load(f)
        count_pixels += a.shape[2]
    print(count_pixels)


if __name__ == '__main__':
    get_shapes(PSE)
# ========================= EOF ====================================================================

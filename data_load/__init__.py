import os
import pickle as pkl

try:
    NORM = pkl.load(open('/media/nvm/ts_data/cm/pixels/meanstd.pkl', 'rb'))
except FileNotFoundError:
    NORM = None
    pass

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

import os
import torch
from torchvision import transforms
from webdataset import dataset as wds

from data_prep import BANDS, CHANNELS, SEQUENCE_LEN


def transform_(x, mean_std):
    mean, std = torch.tensor(mean_std[0]), torch.tensor(mean_std[1])
    normalize = transforms.Normalize(
        mean=mean,
        std=std)
    t = transforms.Compose([normalize])
    x = t(x)
    return x


def image_dataset(mode, data_dir, norm):
    def map_fn(item):
        item = item['pth']
        x = item[:, :, :BANDS]
        x = x.permute(2, 0, 1)
        x = transform_(x, norm)
        x = x.permute(1, 2, 0)
        x = x.reshape(x.shape[0], x.shape[1], SEQUENCE_LEN, CHANNELS).float()
        y = item[:, :, -4:].permute(2, 0, 1)
        g = item[:, :, BANDS:BANDS + 3].permute(2, 0, 1)
        return x, y, g

    loc = os.path.join(data_dir, '{}_patches'.format(mode))
    end_idx = len(os.listdir(loc)) - 1
    brace_str = '{}_{{000000..{}}}.tar'.format(mode, str(end_idx).rjust(6, '0'))
    url = os.path.join(loc, brace_str)
    dataset = wds.Dataset(url).decode('torchl').map(map_fn)
    return dataset


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

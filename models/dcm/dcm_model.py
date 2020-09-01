"""

Credits to  https://github.com/Lab-IDEAS/DeepCropMapping

Deep Crop Mapper module

Credits:
The module from the works of Jinfan Xu et al. on pixel-based crop type mapping.

paper: https://www.sciencedirect.com/science/article/pii/S0034425720303163
code: https://github.com/Lab-IDEAS/DeepCropMapping
"""
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.dcm.dcm_helper import DCMHelper
from models.dcm.utils import PrettyLogger
DEVICE = torch.device("cuda")


logger = PrettyLogger()
helper = DCMHelper()
scaler_ = StandardScaler()


def train(net, x_train, x_test, y_train, y_test):

    train_dataloader = helper.make_data_loader(x_train, y_train, shuffle=True)
    test_dataloader = helper.make_data_loader(x_test, y_test, shuffle=False)

    loss_train_list, acc_train_list, attn_train_list = [], [], []
    loss_test_list, acc_test_list, attn_test_list = [], [], []
    helper.train_model(
        net, train_dataloader, test_dataloader, DEVICE, logger,
        loss_train_list, acc_train_list, attn_train_list,
        loss_test_list, acc_test_list, attn_test_list,
    )

    test_dataloader = helper.make_data_loader(x_test, y_test, shuffle=False)

    y_train_soft_pred, y_train_hard_pred, attn_train = helper.predict(
        net, helper.make_data_loader(x_train, y_train, shuffle=False), DEVICE)

    y_test_soft_pred, y_test_hard_pred, attn_test = helper.predict(
        net, test_dataloader, DEVICE)

    acc_train = accuracy_score(y_train, y_train_hard_pred)
    acc_test = accuracy_score(y_test, y_test_hard_pred)
    logger.info("train acc:", acc_train, "test acc:", acc_test)

    return None


def run_model(data_dir):
    features = None
    np_data = os.path.join(data_dir, 'DATA')
    meta_data = os.path.join(data_dir, 'META')
    _files = [os.path.join(np_data, x) for x in os.listdir(np_data)]
    with open(os.path.join(meta_data, 'labels.json'), 'r') as file:
        label = json.loads(file.read())['label_4class']
    label = [v for k, v in label.items()]
    first = True
    count = 0
    labels = []
    for j, (f, l) in enumerate(zip(_files, label), start=1):
        a = np.load(f)
        labels.extend([l for _ in range(a.shape[-1])])
        count += a.shape[-1]
        if first:
            first = False
            features = a[:, :6, :]
        else:
            features = np.append(features, a[:, :6, :], axis=-1)
        if count > 100000:
            break

    features = np.swapaxes(features, 0, 2)
    features = np.swapaxes(features, 1, 2)
    print(len(labels), features.shape)
    x, x_test, y, y_test = train_test_split(features, labels, test_size=0.33,
                                            random_state=None)

    x = helper.input_x(x)
    y = helper.input_y(y)
    x_test = helper.input_x(x_test)
    y_test = helper.input_y(y_test)

    scaler, x, x_test = helper.normalize_without_scaler(x, x_test)

    net = helper.build_model()
    helper.init_parameters(net)
    net = nn.DataParallel(net)
    net.to(DEVICE)

    train(net, x, x_test, y, y_test)


if __name__ == '__main__':
    path = Path(__file__).parents
    PSE = os.path.join(path[2], 'data', 'pixel_sets')
    run_model(PSE)
# ========================= EOF ====================================================================

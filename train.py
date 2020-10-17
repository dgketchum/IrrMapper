import os
import json
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torchnet as tnt

from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import confusion_matrix_analysis, get_conf_matrix
from models.model_init import get_model
from data_prep.data_loader import get_loaders
from configure import get_config
from utils import recursive_todevice


def train_epoch(model, optimizer, criterion, loader, device, config):
    loss = None
    for i, (x, y, g) in enumerate(loader):
        x = recursive_todevice(x, device)
        if config['model'] == 'clstm':
            y = y.argmax(dim=1).to(device)
            optimizer.zero_grad()
            out, att = model(x)
            pred = out[0][0]
            loss = criterion(pred, y)
        else:
            y = y.to(device)
            optimizer.zero_grad()
            x = torch.squeeze(x)
            out, att = model(x)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        if (i + 1) % config['display_step'] == 0:
            print('Step {}, Loss: {:.4f}'.format(i, loss.item()))

    print('Train Loss: {:.4f}'.format(loss.item()))


def evaluate_epoch(model, criterion, loader, config):
    device = torch.device(config['device'])
    n_class = config['num_classes']
    confusion = torch.tensor(np.zeros((n_class, n_class))).to(device)

    for i, (x, y, g) in enumerate(loader):
        x = recursive_todevice(x, device)
        if config['model'] == 'clstm':
            mask = y.sum(1) > 0
            y = y.argmax(dim=1).to(device)
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                loss = criterion(pred, y)
                pred = torch.argmax(pred, dim=1)
                confusion += get_conf_matrix(y[mask], pred[mask], n_class, device)
        else:
            y = y.squeeze().to(device)
            mask = (y > 0).flatten()
            x = x.squeeze()
            with torch.no_grad():
                pred, att = model(x)
                loss = criterion(pred, y)
                pred = torch.argmax(pred, dim=1)
                conf = get_conf_matrix(y[mask], pred[mask], config['num_classes'], device)
                confusion += conf

    per_class, overall = confusion_matrix_analysis(confusion)
    prec, rec, f1 = overall['precision'], overall['recall'], overall['f1-score']
    print('Evaluation Loss: {:.4f}, Precision {:.4f}, Recall {:.4f}, '
          'F1 Score {:.2f}, \n {}'.format(loss.item(), prec, rec, f1, conf))
    pkl.dump(confusion, open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'wb'))


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['res_dir'], 'figures'), exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'wb'))


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    for fold in range(1, config['kfold'] + 1):
        cm += pkl.load(open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  F1: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def train(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])

    prepare_output(config)
    device = torch.device(config['device'])

    train_loader, test_loader, val_loader = get_loaders(config)
    model = get_model(config)
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = FocalLoss(alpha=config['alpha'], gamma=2, size_average=True)

    # config['N_params'] = model.param_ratio()

    with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as _file:
        _file.write(json.dumps(config, indent=4))

    trainlog = {}

    print('\nTrain {}'.format(config['model'].upper()))
    for epoch in range(1, config['epochs'] + 1):
        print('\nEPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
        model.eval()
        evaluate_epoch(model, criterion, val_loader, config=config)

    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(config['res_dir'], 'model.pth.tar'))

    print('\nRun test set....')
    model.load_state_dict(torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))['state_dict'])
    model.eval()
    evaluate_epoch(model, criterion, test_loader, config=config)

    # overall_performance(config)


if __name__ == '__main__':
    config = get_config('clstm', 'cdl')
    train(config)

# ========================================================================================

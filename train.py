import os
import json
import pickle as pkl
from pathlib import Path

import torch
import numpy as np

from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import confusion_matrix_analysis
from utils import plot_prediction

from models.model_init import get_loaders, get_model

path = Path(__file__).parents


def train_epoch(model, optimizer, criterion, loader, device, config):
    for i, (x, y) in enumerate(loader):

        x = recursive_todevice(x, device)
        mask = y.sum(1) > 0
        y = y.argmax(dim=1).to(device)
        optimizer.zero_grad()
        out, att = model(x)
        pred = out[0][0]
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}'.format(i + 1, len(loader), loss.item()))


def evaluation(model, criterion, loader, device, config, mode='val', plot=False):
    for i, (x, y) in enumerate(loader):
        x = recursive_todevice(x, device)
        y = y.argmax(dim=1).to(device)

        with torch.no_grad():
            out, att = model(x)
            pred = out[0][0]
            loss = criterion(pred, y)
            if mode == 'val':
                pred = torch.argmax(pred, dim=1)
                if plot and i < 5:
                    plot_prediction(pred.cpu().numpy(), y.cpu().numpy())
                iou = intersection_union(pred, y)

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, IoU : {:.2f}'.format(i + 1, len(loader), loss.item(), iou))


def intersection_union(pred, label):
    intersection = (pred & label).float()
    union = (pred | label).float()
    iou = (intersection + 1e-6) / (union + 1e-6)
    iou = iou.mean().item()
    return iou


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    for fold in range(1, config['kfold'] + 1):
        os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)


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
        cm += pkl.load(open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)

    device = torch.device(config['device'])

    train_loader, test_loader, val_loader = get_loaders(config)
    print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

    model = get_model(config)

    # config['N_params'] = model.param_ratio()

    with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as _file:
        _file.write(json.dumps(config, indent=4))

    model = model.to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = FocalLoss(alpha=config['alpha'], gamma=2, size_average=True)

    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
        print('Validation . . . ')
        model.eval()
        evaluation(model, criterion, val_loader, device=device, config=config, mode='val', plot=False)

        # trainlog[epoch] = {**train_metrics, **val_metrics}
        # checkpoint(trainlog, config)

    evaluation(model, criterion, val_loader, device=device, config=config, mode='val', plot=True)
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(config['res_dir'], 'model.pth.tar'))

    print('Testing best epoch . . .')
    model.load_state_dict(torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))['state_dict'])
    model.eval()

    # test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)
    # save_results(test_metrics, conf_mat, config)

    overall_performance(config)


if __name__ == '__main__':

    config = {'mode': 'irr',
              'rdm_seed': 1,
              'epochs': 1,
              'display_step': 10,
              'num_classes': 4,
              'nomenclature': 'label_4class',
              'kfold': 5,
              'input_dim': 7,
              'geomfeat': None,
              'device': 'cuda:0',
              'num_workers': 4,
              'pooling': 'mean_std',
              'dropout': 0.2,
              'gamma': 1,
              'alpha': None,
              'validation_folder': os.path.join(path[0], 'data', 'npy', 'valid'),
              'ltae': False, 'dcm': False, 'tcnn': False, 'clstm': True}

    if config['ltae']:
        config['dataset_folder'] = os.path.join(path[0], 'data', 'pixel_sets')
        config['batch_size'] = 128
        config['mlp1'] = '[7, 32, 64]'
        config['mlp2'] = '[131, 128]'
        config['mlp3'] = '[256,128]'
        config['mlp4'] = '[128, 64, 32, 20]'
        config['n_head'] = 16
        config['d_k'] = 8
        config['d_model'] = 256
        config['T'] = 1000
        config['positions'] = None
        config['geom_dim'] = 3
        config['geomfeat'] = True
        config['lms'] = 13
        config['npixel'] = 64
        config['subset'] = None
        config['lr'] = 0.00025
        config['num_classes'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'ltae_pse', 'results')
        with open(os.path.join(path[0], 'models', 'ltae_pse', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))
            # exit()

    if config['dcm']:
        config['batch_size'] = 7168
        config['dataset_folder'] = os.path.join(path[0], 'data', 'pixels')
        config['hidden_size'] = 256
        config['num_layers'] = 2
        config['bidirectional'] = True
        config['seed'] = 121
        config['lr'] = 0.00025
        config['res_dir'] = os.path.join(path[0], 'models', 'dcm', 'results')
        with open(os.path.join(path[0], 'models', 'dcm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['tcnn']:
        config['batch_size'] = 7168
        config['dataset_folder'] = os.path.join(path[0], 'data', 'pixels')
        config['sequence_len'] = 13
        config['nker'] = '[16, 16, 16]'
        config['mlp3'] = '[16, 16]'
        config['res_dir'] = os.path.join(path[0], 'models', 'temp_cnn', 'results')
        with open(os.path.join(path[0], 'models', 'temp_cnn', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    if config['clstm']:
        config['batch_size'] = 8
        config['input_dim'] = 7
        config['num_layers'] = 1
        config['dataset_folder'] = os.path.join(path[0], 'data', 'npy')
        config['kernel_size'] = (3, 3)
        config['hidden_dim'] = 4
        config['res_dir'] = os.path.join(path[0], 'models', 'conv_lstm', 'results')
        with open(os.path.join(path[0], 'models', 'conv_lstm', 'config.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))

    for k, v in config.items():
        if 'mlp' in k or k == 'nker':
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    # pprint.pprint(config)
    main(config)

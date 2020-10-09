import os
import json
import pickle as pkl

import torch
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss

from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import confusion_matrix_analysis
from models.model_init import get_loaders, get_model
from configure import get_config
from utils import recursive_todevice


def train_epoch(model, optimizer, criterion, loader, device, config):
    for i, (x, y) in enumerate(loader):
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
            loss = criterion(out, y.long())

        loss.backward()
        optimizer.step()
    print('Step {}, Loss: {:.4f}'.format(i + 1, loss.item()))


def evaluate_epoch(model, criterion, loader, device, config):
    confusion = torch.ones_like(config['confusion'])
    for i, (x, y) in enumerate(loader):
        x = recursive_todevice(x, device)

        if config['model'] == 'clstm':
            mask = y.sum(1) > 0
            y = y.argmax(dim=1).to(device)
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                loss = criterion(pred, y)
                pred = torch.argmax(pred, dim=1)
                confusion += conf_matrix(y, pred, config, mask)
        else:
            y = y.to(device)
            x = x.squeeze()
            with torch.no_grad():
                pred, att = model(x)
                loss = criterion(pred, y)
                pred = torch.argmax(pred, dim=1)
                confusion += conf_matrix(y, pred, config)

    per_class, overall = confusion_matrix_analysis(confusion)
    prec, rec, f1 = overall['micro_Precision'], overall['micro_Recall'], overall['micro_F1-score']
    print('Step {}, Loss: {:.4f}, Precision {:.2f}, Recall {:.2f}, '
          'F1 Score {:.2f},'.format(i + 1, loss.item(), prec, rec, f1))


def conf_matrix(y, pred, config, mask=None):
    confusion = torch.ones_like(config['confusion'])
    n_classes = config['num_classes']
    classes = torch.tensor([x for x in range(n_classes)]).to(torch.device(config['device']))
    if mask:
        y, pred = y[mask], pred[mask]
    for i in range(n_classes):
        c = classes[i]
        pred, target = pred == c, y == c
        confusion[i, i] = (pred & target).bool().sum()
        for nc in range(n_classes):
            if nc == c:
                continue
            else:
                confusion[c, nc] = (pred == nc).bool().sum()
    config['confusion'] += confusion
    return confusion


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
    # print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

    model = get_model(config)

    # config['N_params'] = model.param_ratio()

    with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as _file:
        _file.write(json.dumps(config, indent=4))

    model = model.to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = FocalLoss(alpha=config['alpha'], gamma=2, size_average=True)
    # criterion = CrossEntropyLoss()
    config['confusion'] = torch.tensor(np.zeros((config['num_classes'], config['num_classes']))).to(device)

    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)
        print('Validation . . . ')
        model.eval()
        evaluate_epoch(model, criterion, val_loader, device=device, config=config)

        # trainlog[epoch] = {**train_metrics, **val_metrics}
        # checkpoint(trainlog, config)

    evaluate_epoch(model, criterion, val_loader, device=device, config=config)
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
    config = get_config()
    train(config)

# ========================================================================================

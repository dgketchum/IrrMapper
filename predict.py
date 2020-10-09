import os
import torch

from matplotlib import pyplot as plt

from utils import recursive_todevice
from models.model_init import get_model, get_predict_loader
from configure import get_config


def predict(config):
    device = torch.device(config['device'])
    val_loader = get_predict_loader(config)
    print('Predict')
    model = get_model(config)
    check_pt = torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(check_pt['state_dict'])
    model.to(device)
    optimizer.load_state_dict(check_pt['optimizer'])
    model.eval()
    prediction(model, val_loader, device, config)


def prediction(model, loader, device, config):
    print('writing predictions to {}'.format(config['res_dir']))
    for i, (x, y) in enumerate(loader):
        print(x.shape)
        x = recursive_todevice(x, device)
        if config['model'] == 'clstm':
            y = y.argmax(dim=1).to(device)
            with torch.no_grad():
                out, att = model(x)
                pred = out[0][0]
                pred = torch.argmax(pred, dim=1)
        else:
            y = y.to(device)
            x = x.squeeze()
            with torch.no_grad():
                pred, att = model(x)
                pred = torch.argmax(pred, dim=1)

        plot_prediction(pred.cpu().numpy(), y.cpu().numpy(),
                        os.path.join(config['res_dir'], 'figures', '{}.png'.format(i)))


cmap = ListedColormap(['grey', 'blue', 'purple', 'pink', 'green'])


def plot_prediction(pred, label, out_file=None):
    fig, ax = plt.subplots(ncols=2, nrows=8, figsize=(10, 40))
    for i in range(8):
        ax[i, 0].imshow(pred[i], cmap=cmap)
        ax[i, 1].imshow(label[i], cmap=cmap)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    config = get_config()
    predict(config)
# ========================= EOF ====================================================================

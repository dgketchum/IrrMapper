"""
Credits to  https://github.com/VSainteuf/pytorch-psetae
author: Vivien Sainte Fare Garnot
paper: https://arxiv.org/pdf/2007.00586.pdf
"""
import torch.nn as nn

from models.ltae_pse.pse import PixelSetEncoder
from models.ltae_pse.ltae import LTAE


class PseLTae(nn.Module):
    """
    Pixel-Set encoder + Lightweight Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
                 n_head=16, d_k=8, d_model=256, mlp3=[256, 128], dropout=0.2, T=1000, len_max_seq=24, positions=None,
                 mlp4=[128, 64, 32, 20], return_att=False):
        super(PseLTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = LTAE(in_channels=mlp2[-1], n_head=n_head, d_k=d_k,
                                     d_model=d_model, n_neurons=mlp3, dropout=dropout,
                                     T=T, len_max_seq=len_max_seq, positions=positions, return_att=return_att
                                     )
        self.decoder = get_decoder(mlp4)
        self.return_att = return_att

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)
        if self.return_att:
            out, att = self.temporal_encoder(out)
            out = self.decoder(out)
            return out, att
        else:
            out = self.temporal_encoder(out)
            out = self.decoder(out)
            return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

        return total


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_decoder(n_neurons):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
    """
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
        if i < (len(n_neurons) - 2):
            layers.extend([
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m

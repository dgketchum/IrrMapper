"""

Credits to  https://github.com/Lab-IDEAS/DeepCropMapping

Deep Crop Mapper module

Credits:
The module from the works of Jinfan Xu et al. on pixel-based crop type mapping.

paper: https://www.sciencedirect.com/science/article/pii/S0034425720303163
code: https://github.com/Lab-IDEAS/DeepCropMapping
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCM(nn.Module):
    def __init__(
            self, seed, input_dim, hidden_size, num_layers,
            bidirectional, dropout, num_classes
    ):
        super().__init__()
        self._set_reproducible(seed)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )  # i/o: (batch, seq_len, num_directions*input_/hidden_size)
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        self.fc = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=num_classes,
        )

    def _set_reproducible(self, seed, cudnn=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        self.lstm.flatten_parameters()
        # lstm_out: (batch, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        # softmax along seq_len axis
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), dim=1)
        # attn (after permutation): (batch, 1, seq_len)
        fc_in = attn_weights.permute(0, 2, 1).bmm(lstm_out)
        fc_out = self.fc(fc_in)
        fc_out = fc_out.squeeze()
        fc_out = F.softmax(fc_out, dim=1)
        return fc_out, attn_weights.squeeze()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

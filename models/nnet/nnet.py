import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out, None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

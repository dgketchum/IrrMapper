import torch


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================

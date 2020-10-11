import numpy as np
import torch


def get_conf_matrix(y, pred, n_class, device):
    batch_conf = torch.tensor(np.zeros((n_class, n_class))).to(device)
    classes = torch.tensor([x for x in range(n_class)]).to(device)
    for i in range(n_class):
        c = classes[i]
        pred, target = pred == c, y == c
        batch_conf[i, i] = (pred & target).bool().sum()
        for nc in range(n_class):
            if nc != c:
                batch_conf[c, nc] = (pred == nc).bool().sum()
        return batch_conf


def confusion_matrix_analysis(mat):
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = torch.sum(mat[j, j])
        fp = torch.sum(mat[:, j]) - tp
        fn = torch.sum(mat[j, :]) - tp

        d['IoU'] = tp / (tp + fp + fn)
        d['Precision'] = tp / (tp + fp)
        d['Recall'] = tp / (tp + fn)
        d['F1-score'] = 2 * tp / (2 * tp + fp + fn)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn
    overall = {'IoU': TP / (TP + FP + FN),
               'precision': TP / (TP + FP),
               'recall': TP / (TP + FN),
               'f1-score': 2 * TP / (2 * TP + FP + FN),
               'accuracy': torch.sum(torch.diag(mat)) / torch.sum(mat)}

    return per_class, overall

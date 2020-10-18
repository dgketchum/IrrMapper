import numpy as np
import torch


def get_conf_matrix(y, pred, n_class, device):
    batch_conf = torch.tensor(np.zeros((n_class, n_class))).to(device)
    for i in range(len(y)):
        batch_conf[y[i]][pred[i]] += 1
    return batch_conf


def confusion_matrix_analysis(mat):
    epsilon = 1.0e-6
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = torch.sum(mat[j, j])
        fp = torch.sum(mat[:, j]) - tp
        fn = torch.sum(mat[j, :]) - tp

        d['iou'] = tp / (tp + fp + fn + epsilon)
        d['precision'] = tp / (tp + fp + epsilon)
        d['recall'] = tp / (tp + fn + epsilon)
        d['f1-score'] = 2 * tp / (2 * tp + fp + fn + epsilon)

        per_class[str(j)] = {k: v.item() for k, v in d.items()}

        TP += tp
        FP += fp
        FN += fn

    overall = {'iou': TP / (TP + FP + FN + epsilon),
               'precision': TP / (TP + FP + epsilon),
               'recall': TP / (TP + FN + epsilon),
               'f1-score': 2 * TP / (2 * TP + FP + FN + epsilon),
               'accuracy': torch.sum(torch.diag(mat)) / torch.sum(mat)}

    overall = {k: v.item() for k, v in overall.items()}

    return per_class, overall


if __name__ == '__main__':
    pass
# =======================================================================================

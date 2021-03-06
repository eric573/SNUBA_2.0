import numpy as np
from sklearn.metrics import f1_score

def apply_threshold(y_prob, beta):
    # Label data
    idx_pos = np.where(y_prob >= 0.5 + beta)
    idx_zero = np.where(np.abs(y_prob - 0.5) < beta)
    idx_neg = np.where(y_prob <= 0.5 - beta)
    y_hat = np.zeros(y_prob.shape[0])
    y_hat[idx_neg] = -1
    y_hat[idx_zero] = 0
    y_hat[idx_pos] = 1
    return y_hat

def calcF1(y_star, y_hat):
    # F1 calculation
    var = np.sum(np.abs(y_hat))
    precision = (np.sum(y_hat == y_star) / var) if var != 0 else 0
    recall = np.sum(y_hat == y_star) / len(y_hat)
    if recall + precision == 0:
        # print(f1_score(y_star, y_hat, average='micro'))
        return 0
    # print(f1_score(y_star, y_hat, average='micro'), 2 * (precision * recall) / (precision + recall))
    return 2 * (precision * recall) / (precision + recall)

def calcJaccard(y_U, n, beta):
    idx = np.where(np.abs(y_U - 0.5) >= beta)
    n_idx = np.where(n == 1)
    intersect = np.sum(np.in1d(idx[0], n_idx[0]))
    union = len(np.union1d(idx[0], n_idx[0]))
    # print(1 - (intersect / union), intersect, union)
    return 1 - (intersect / union), idx


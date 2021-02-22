import numpy as np

def calcF1(y_star, y_prob, beta):
    # Label data
    idx_pos = np.where(y_prob >= 0.5 + beta)
    idx_zero = np.where(np.abs(y_prob - 0.5) < beta)
    idx_neg = np.where(y_prob <= 0.5 - beta)
    y_hat = np.zeros(y_prob.shape[0])
    y_hat[idx_neg] = -1
    y_hat[idx_zero] = 0
    y_hat[idx_pos] = 1

    # F1 calculation
    var = np.sum(np.abs(y_hat))
    precision = (np.sum(y_hat == y_star) / var) if var != 0 else 0
    recall = np.sum(y_hat == y_star) / len(y_hat)
    if recall + precision == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calcJaccard(y_U, n, beta):
    idx = np.where(np.abs(y_prob - 0.5) >= beta)
    intersect = len(np.in1d(idx, n))
    union = len(np.union1d(idx, n))
    return 1 - (intersect / union)


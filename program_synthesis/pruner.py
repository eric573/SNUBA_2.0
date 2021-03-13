import numpy as np
from program_synthesis.utils import calcF1, calcJaccard, apply_threshold

class Pruner(object):
    def __init__(self, H, H_C, y_star, primitive_XL, primitive_XU, n, w, X_comb_idx):
        self.H = H
        self.H_C = H_C
        self.y_star = y_star
        self.primitive_XL = primitive_XL # TODO: code cleanup - we use this in test.py
        self.primitive_XU = primitive_XU #TODO: code cleanup - we use this in test.py
        self.n = n
        self.w = w
        self.X_comb_idx = X_comb_idx

    def prune(self, basecase=True):
        h_best = []

        for ((h, beta), X_comb) in zip(self.H, self.X_comb_idx):
            y_L = h.predict_proba(self.primitive_XL[:, np.array(X_comb)])[:, 1]
            y_prob = apply_threshold(y_L, beta)
            f_score = calcF1(self.y_star, y_prob)

            y_U = h.predict_proba(self.primitive_XU[:, np.array(X_comb)])[:, 1]
            j_score, idx = calcJaccard(y_U, self.n, beta)

            h_best.append(((h, beta), idx, y_prob, X_comb, (1 - self.w) * j_score + self.w * f_score))
        h_best.sort(key=lambda x: -x[-1])
        for i in range(3 if basecase else 1):
            self.n[h_best[i][1]] = 1

        return h_best

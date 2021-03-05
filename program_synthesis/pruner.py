import numpy as np
from program_synthesis.utils import calcF1, calcJaccard

class Pruner(object):
    def __init__(self, H, H_C, y_star, primitive_XL, primitive_XU, n, w, X_comb_idx):
        self.H = H
        self.H_C = H_C # TODO: Add h_best to H_C
        self.y_star = y_star
        self.primitive_XL = primitive_XL
        self.primitive_XU = primitive_XU
        self.n = n
        self.w = w
        self.X_comb_idx = X_comb_idx

    def prune(self):
        h_best = None
        bestScore = 0
        best_idx = None
        for ((h, beta), X_comb) in zip(self.H, self.X_comb_idx):
            y_L = h.predict_proba(self.primitive_XL[:, np.array(X_comb)])[:, 1]
            f_score = calcF1(self.y_star, y_L, beta)

            y_U = h.predict_proba(self.primitive_XU[:, np.array(X_comb)])[:, 1]
            j_score, idx = calcJaccard(y_U, self.n, beta)

            if self.w * (j_score + f_score) >= bestScore:
                h_best = (h, beta)
                bestScore = (1 - self.w) * j_score + self.w * f_score
                best_idx = idx

        self.n[best_idx] = 1
        return h_best

import numpy as np
from program_synthesis.utils import calcF1, calcJaccard, apply_threshold

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
        # TODO: Keep best 3 heuristics 
        
        h_best = None
        bestScore = 0
        best_idx = None
        best_y_prob = None
        best_X_comb = None
        for ((h, beta), X_comb) in zip(self.H, self.X_comb_idx):
            y_L = h.predict_proba(self.primitive_XL[:, np.array(X_comb)])[:, 1]
            y_prob = apply_threshold(y_L, beta)
            f_score = calcF1(self.y_star, y_prob)

            y_U = h.predict_proba(self.primitive_XU[:, np.array(X_comb)])[:, 1]
            j_score, idx = calcJaccard(y_U, self.n, beta)

            if self.w * (j_score + f_score) >= bestScore:
                h_best = (h, beta)
                bestScore = (1 - self.w) * j_score + self.w * f_score
                best_idx = idx
                best_y_prob = y_prob
                best_X_comb = X_comb
        self.n[best_idx] = 1
        return h_best, best_X_comb, best_y_prob

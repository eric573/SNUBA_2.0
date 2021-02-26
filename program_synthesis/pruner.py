import numpy as np
from utils import calcF1

class Pruner(object):
    def __init__(self, H, H_C, y_star, primitive_XL, primitive_XU, n, w):
        self.H = H
        self.H_C = H_C # TODO: Add h_best to H_C 
        self.y_star = y_star
        self.primitive_XL = primitive_XL
        self.primitive_XU = primitive_XU
        self.n = n
        self.w = w

    def prune(self):
        h_best = None
        bestScore = 0
        for (h, beta) in self.H:
            y_L = h.predict(self.primitive_XL)
            f_score = calcF1(self.y_star, y_L, beta)

            y_U = h.predict(self.primitive_XU)
            j_score = calcJaccard(y_U, self.n)

            if w * (j_score + f_score) >= bestScore:
                h_best = (h, beta)
                bestScore = (1 - w) * j_score + w * f_score

        return h_best

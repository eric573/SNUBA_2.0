import numpy as np
from label_aggregator import LabelAggregator
import math
from utils import calcF1

eta = 3/2
delta = 0.001

# def calcAcc(H_C, primitive_XL, y_star):
#     alpha_hat = []
#     for (h, beta) in H_C:
#         y_L_i = h.predict(primitive_XL)
#         N_i = np.count_nonzero(y_L)
#         alpha_i = 1/N_i * np.sum(y_L_i == y_star)
#         alpha_hat.append(alpha_i)
#
#     return np.array(alpha_hat)

def findNu(M):
    """
    M is the number of heuristics in the committed set
    """
    # WARNING: the paper said M+1, but their code is only M.
    return 1/2 - 1/((M+1) ** eta)

class Verifier(object):
    def __init__(self, H, H_C, y_star, primitive_XL, primitive_XU, n, w):
        self.H = H
        self.H_C = H_C
        self.y_star = y_star
        self.primitive_XL = primitive_XL
        self.primitive_XU = primitive_XU
        self.n = n
        self.w = w

        self.gen_model = None


    def train_gen_mode(self):
        gen_model = LabelAggregator()
        # the generative model does NOT use labels when training
        gen_model.train(self.primitive_XU)
        self.gen_model = gen_model


    def verify(self):
        if self.gen_model is None:
            self.train_gen_mode()
        else:
            print('WARNING: using existing generative model. Maybe doing something wrong!')

        self.y_tilde_U = self.gen_model.marginals(self.primitive_XU)
        self.y_tilde_L = self.gen_model.marginals(self.primitive_XL)

    def find_vague_points(self, nu=0.1):
        """
        nu is synonym for gamma in their code
        use findNu for accurate nu
        """
        return self.y_tilde_L[np.where(self.y_tilde_L - 0.5) <= nu]
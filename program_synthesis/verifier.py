import numpy as np
from utils import calcF1

eta = 3/2
delta = 0.001

def learnAcc(H_C, primitive_XU):
    # TODO
    return

def calcAcc(H_C, primitive_XL, y_star):
    alpha_hat = []
    for (h, beta) in H_C:
        y_L_i = h.predict(primitive_XL)
        N_i = np.count_nonzero(y_L)
        alpha_i = 1/N_i * np.sum(y_L_i == y_star)
        alpha_hat.append(alpha_i)

    return np.array(alpha_hat)

def calcLabels(alpha_tilde, primitive_X): # generative model
    # TODO
    return

def findEps(N_U, M):
    # TODO
    return gamma - (1/(2 * N) * math.log(2*M/delta)) ** (1/2)

def findNu(M):
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

    def verify(self):
        alpha_tilde = learnAcc(self.H_C, self.primitive_XU)
        alpha_hat = calcAcc(self.H_C, self.primitive_XL, self.y_star)
        y_tilde_U = calcLabels(alpha_tilde, self.primitive_XU)
        y_tilde_L = calcLabels(alpha_tilde, self.primitive_XL)
        N_U = self.primitive_XU.shape[0]
        # M is the number of heuristics in the committed set
        M = len(self.H_C)
        epsilon = findEps(N_U, M)
        nu = findNu(M)
        if np.linalg.norm(alpha_tilde - alpha_hat, ord=np.inf) >= epsilon:
            return set(), y_tilde_U
        else:
            O_prime_L = set()

            for i in range(len(y_tilde_L)): # TODO: Vectorize
                if math.abs(y_tilde_L[i] - 0.5) <= nu:
                    O_prime_L.add(self.primitive_XL[i])

            return O_prime_L, y_tilde_U
import itertools
import numpy as np
from tqdm import tqdm, trange
np.seterr('raise')

class Synthesizer(object):
    def __init__(self, model, primitive_X, labels):
        self.model = model
        self.primitive_X = primitive_X
        self.labels = labels


    def solve(self):
        H = []
        X_comb = []

        """
            Assume primitive_X is (n x d)
        """
        print(self.primitive_X.shape[1])
        for D_prime in range(1, min(4, self.primitive_X.shape[1] + 1)):
            idx_comb = itertools.combinations(range(self.primitive_X.shape[1]), D_prime)
            for comb in tqdm(idx_comb):
                X_prime = self.primitive_X[:,np.array(comb)]
                h = self.model.fit(X_prime, self.labels)
                y_prob = self.predictProb(h, X_prime)[:, 1]
                beta = self.findBeta(y_prob, self.labels)
                H.append((h, beta))
                X_comb.append(X_prime)
        return H, X_comb

    def predictProb(selfs, heuristic_model, X_prime):
        return heuristic_model.predict_proba(X_prime)

    def findBeta(self, y_prob, label):
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


        beta_list = np.arange(0, 0.55, 0.05)
        f1 = np.zeros(len(beta_list))
        for j in range(len(beta_list)):
            beta = beta_list[j]
            f1[j] = calcF1(label, y_prob, beta)
        return beta_list[np.argmax(f1)]


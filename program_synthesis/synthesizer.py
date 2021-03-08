import itertools
import numpy as np
from tqdm import tqdm, trange
from program_synthesis.utils import calcF1, apply_threshold
np.seterr('raise')

class Synthesizer(object):
    def __init__(self, model, primitive_X, labels, min_D = 2):
        self.model = model
        self.primitive_X = primitive_X

        self.labels = labels

        # D_prime would be "number of features choose min_D"
        # + 1 to be inclusive
        self.min_D = min_D + 1

    def solve(self, idx=None):
        H = []
        X_comb = []

        if idx is not None:
            primitive_X_new = self.primitive_X[idx, :]
            labels_new = self.labels[idx]
        else:
            primitive_X_new = self.primitive_X.copy()
            labels_new = self.labels.copy()


        """
            Assume primitive_X is (n x d)
            
            Returns:
                H: list of (h, b) where h is the heuristics and b is the beta
                X_comb: list of combinations of features used by each heuristics
        """
        print('{} choose {}'.format(primitive_X_new.shape[1], self.min_D - 1))
        for D_prime in range(1, min(self.min_D, primitive_X_new.shape[1] + 1)):
            idx_comb = itertools.combinations(range(primitive_X_new.shape[1]), D_prime)
            for comb in idx_comb:
                X_prime = primitive_X_new[:,np.array(comb)]
                # create a new model every time
                h = self.model()
                h = h.fit(X_prime, labels_new)
                y_prob = self.predictProb(h, X_prime)[:, 1]
                beta = self.findBeta(y_prob, labels_new)
                H.append((h, beta))
                X_comb.append(comb)
        return H, X_comb

    def predictProb(selfs, heuristic_model, X_prime):
        return heuristic_model.predict_proba(X_prime)

    def findBeta(self, y_prob, label):
        beta_list = np.arange(0, 0.55, 0.05)
        f1 = np.zeros(len(beta_list))
        for j in range(len(beta_list)):
            beta = beta_list[j]
            f1[j] = calcF1(label, apply_threshold(y_prob, beta))
        return beta_list[np.argmax(f1)]


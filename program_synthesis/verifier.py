import numpy as np
from program_synthesis.label_aggregator import LabelAggregator
from program_synthesis.utils import *
from sklearn.naive_bayes import BernoulliNB


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
class Verifier(object):
    def __init__(self, H_C, y_star, primitive_XL_Label, primitive_XU_Label, n, w, use_sklearn=False):
        self.H_C = H_C # Commited set
        self.y_star = y_star # True label of the training set -> (-1,1)
        self.primitive_XL_Label = primitive_XL_Label # Labeled training dataset (Features)
        self.primitive_XU_Label = primitive_XU_Label # Unlabeled training dataset (Features)
        self.n = n
        self.w = w

        self.gen_model = None

        self.use_sklearn = use_sklearn


    def train_gen_mode(self):
        if self.use_sklearn:
            gen_model = BernoulliNB()
            # they didn't use the y_stars, but why not? lol
            gen_model.fit(self.primitive_XL_Label, self.y_star)
            self.gen_model = gen_model
        else:
            gen_model = LabelAggregator()
            # the generative model does NOT use labels when training
            gen_model.train(self.primitive_XL_Label)
            self.gen_model = gen_model


    def verify(self):
        if self.gen_model is None:
            self.train_gen_mode()
        else:
            print('WARNING: using existing generative model. Maybe doing something wrong!')

        if self.use_sklearn:
            self.y_tilde_U = self.gen_model.predict_proba(self.primitive_XU_Label)
            self.y_tilde_L = self.gen_model.predict_proba(self.primitive_XL_Label)
            print('y_tilde_U shape', self.y_tilde_U.shape)
            print('primitive_XU_Label shape', self.primitive_XU_Label.shape)
            # self.y_tilde_U[np.where(self.primitive_XU_Label) == 0] = 0.5
            # self.y_tilde_L[np.where(self.primitive_XL_Label) == 0] = 0.5

            # print('y_tilde_U')
            # print(self.y_tilde_U)
            # print('y_tilde_L')
            # print(self.y_tilde_L)
        else:
            self.y_tilde_U = self.gen_model.marginals(self.primitive_XU_Label)
            self.y_tilde_L = self.gen_model.marginals(self.primitive_XL_Label)


    def get_uncertain_points(self, nu=0.1):
        """
        nu is synonym for gamma in their code
        use findNu for accurate nu
        """
        uncertain_idx = np.where(np.abs(self.y_tilde_L - 0.5) <= nu)[0]
        if len(uncertain_idx) == 0:
            return []
        return uncertain_idx

    def update_n(self, nu=0.1):
        uncertain_idx = np.where(np.abs(self.y_tilde_U - 0.5) >= nu)[0]
        return uncertain_idx

    def evaluate(self, train_ground, val_ground):
        """
        Evaluate accuracy and coverage for (y_tilde_U, train_ground) (y_tilde_L, val_ground)
        """
        self.val_accuracy = calculate_accuracy(self.y_tilde_L, val_ground)
        self.train_accuracy = calculate_accuracy(self.y_tilde_U, train_ground)
        self.val_coverage = calculate_coverage(self.y_tilde_L)
        self.train_coverage = calculate_coverage(self.y_tilde_U)
        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage


    def findNu(self, M):
        """
        M is the number of heuristics in the committed set
        """
        # WARNING: the paper said M+1, but their code is only M.
        return 1/2 - 1/((M+1) ** eta)

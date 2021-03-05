import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.utils import calcF1
from program_synthesis.pruner import Pruner

from data.loader import DataLoader
loader = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = loader.load_data(None, "./data/imdb/")

model = sklearn.linear_model.LogisticRegression
synthesizer = Synthesizer(model, train_primitive_matrix, train_ground, 1)
H, X_comb = synthesizer.solve()

H_C = []
n = np.zeros(len(val_primitive_matrix))
w = 0.5

pruner = Pruner(H, H_C, train_ground, train_primitive_matrix, val_primitive_matrix, n, w, X_comb)
h_best = pruner.prune()



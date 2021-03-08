import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from program_synthesis.synthesizer import Synthesizer
from program_synthesis.utils import calcF1, apply_threshold
from program_synthesis.pruner import Pruner
from program_synthesis.verifier import Verifier

from data.loader import DataLoader
basecase = True

loader = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = loader.load_data(None, "./data/imdb/")

train_prob_labels = np.array([])
val_prob_labels = np.array([])
H_C = []
n = np.zeros(len(train_primitive_matrix))
w = 0.5
idx = None

for i in range(25):
    print(f"Iter {i}")
    model = sklearn.linear_model.LogisticRegression
    synthesizer = Synthesizer(model, val_primitive_matrix, val_ground, min_D=2)
    H, X_comb = synthesizer.solve(idx=idx)

    # No shrinking of the val primitive matrix after getting uncertain index
    pruner = Pruner(H, H_C, val_ground, val_primitive_matrix, train_primitive_matrix, n, w, X_comb)
    h_best_context = pruner.prune(basecase=basecase)

    for i in range(3 if basecase else 1):
        h_best = h_best_context[i][0][0]
        X_comb_best = h_best_context[i][3]
        beta_best = h_best_context[i][0][1]
        val_prob_temp_labels = h_best_context[i][2]

        H_C.append(h_best)
        y_U = h_best.predict_proba(train_primitive_matrix[:, np.array(X_comb_best)])[:, 1]
        train_prob_temp_labels = apply_threshold(y_U, beta_best)

        if len(train_prob_labels) == 0:
            train_prob_labels = np.append(train_prob_labels, train_prob_temp_labels)
            train_prob_labels = np.reshape(train_prob_labels, (train_prob_temp_labels.shape[0], 1))
            val_prob_labels = np.append(val_prob_labels, val_prob_temp_labels)
            val_prob_labels = np.reshape(val_prob_labels, (val_prob_temp_labels.shape[0], 1))
        else:
            train_prob_labels = np.concatenate((train_prob_labels, train_prob_temp_labels.reshape((-1, 1))), axis=1)
            val_prob_labels = np.concatenate((val_prob_labels, val_prob_temp_labels.reshape((-1, 1))), axis=1)

    verifier = Verifier(H_C, val_ground, val_prob_labels, train_prob_labels, n, w)
    verifier.verify()
    idx = verifier.get_uncertain_points(nu=verifier.findNu(len(H_C)))
    basecase = False
    print(f"Uncerntain points count: {len(idx)}")
    if len(idx) <= 1:
        print("STOP")
        break



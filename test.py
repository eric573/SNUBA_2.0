import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree

from program_synthesis.synthesizer import Synthesizer
from program_synthesis.utils import calcF1, apply_threshold
from program_synthesis.pruner import Pruner
from program_synthesis.verifier import Verifier

from joblib import dump
from data.loader import DataLoader


'''
We and the Snuba code treat the validation data as X_L (labeled) and training data as X_U (unlabeled)
'''
loader = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = loader.load_data(None, "./data/imdb/")

train_prob_labels = np.array([])
val_prob_labels = np.array([])
H_C = []
n = np.zeros(len(train_primitive_matrix)) # n[i] is 1 if we confidently label example i of X_U
w = 0.5 # weighting for f1/Jaccard dissimilarity in pruner
idx = None
basecase = True # initial run requires selecting more than 1 heuristic. TODO: why?

for step in range(20):
    print(f"Iter {step}")
    # model = sklearn.linear_model.LogisticRegression #TODO: change to multiple heuristic models

    '''
    model should be one of Logistic Regression, DecisionTree, KNNs
    How to choose which one?
    '''

    models = [
        sklearn.linear_model.LogisticRegression,
    #    sklearn.neighbors.KNeighborsClassifier,
    #    sklearn.tree.DecisionTreeClassifier
    ]

    # Synthesizer
    synthesizer = Synthesizer(models, val_primitive_matrix, val_ground, min_D=1)
    H, X_comb = synthesizer.solve(idx=idx)

    # Pruner
    # No shrinking of the val primitive matrix after getting uncertain index
    pruner = Pruner(H, H_C, val_ground, val_primitive_matrix, train_primitive_matrix, n, w, X_comb)
    h_best_context = pruner.prune(basecase=basecase)

    for i in range(3 if basecase else 1):
        # Extract heuristic model, beta, data used, and labels assigned by output of pruner
        h_best = h_best_context[i][0][0]
        X_comb_best = h_best_context[i][3]
        beta_best = h_best_context[i][0][1]
        val_prob_temp_labels = h_best_context[i][2]
        H_C.append(h_best)

        # Run pruner output on unlabeled dataset
        y_U = h_best.predict_proba(train_primitive_matrix[:, np.array(X_comb_best)])[:, 1]
        train_prob_temp_labels = apply_threshold(y_U, beta_best)

        # Aggregating the labels the heuristic gives for the labeled and unlabeled datasets
        if len(train_prob_labels) == 0:
            train_prob_labels = np.append(train_prob_labels, train_prob_temp_labels)
            train_prob_labels = np.reshape(train_prob_labels, (train_prob_temp_labels.shape[0], 1))
            val_prob_labels = np.append(val_prob_labels, val_prob_temp_labels)
            val_prob_labels = np.reshape(val_prob_labels, (val_prob_temp_labels.shape[0], 1))
        else:
            train_prob_labels = np.concatenate((train_prob_labels, train_prob_temp_labels.reshape((-1, 1))), axis=1)
            val_prob_labels = np.concatenate((val_prob_labels, val_prob_temp_labels.reshape((-1, 1))), axis=1)

    # Verifier
    verifier = Verifier(H_C, val_ground, val_prob_labels, train_prob_labels, n, w)
    verifier.verify()
    idx = verifier.get_uncertain_points(nu=verifier.findNu(len(H_C)))

    # Evaluate
    val_accuracy, train_accuracy, val_coverage, train_coverage = verifier.evaluate(train_ground, val_ground)
    print("Train: Accuracy {}. Coverage {}.".format(train_accuracy, train_coverage))
    print("Val: Accuracy {}. Coverage {}.".format(val_accuracy, val_coverage))

    basecase = False
    print(f"Uncertain points count: {len(idx)}")
    if len(idx) <= 1:
        print("STOP")
        break
    print()

# persist models to memory bois
dump(H_C, 'models.joblib')
print('heuristics saved')

# persist the labels
dump(verifier.y_tilde_U, 'train_prob_labels.joblib')
print('train prob labels saved')

from program_synthesis.synthesizer import Synthesizer
from data.loader import DataLoader
import sklearn.linear_model

"""
Input:  Heuristic Models, M
        Primitive Matrix, X
        Labels y*

Output: Candidate set of heuristic H
        Primitive combinations X_comb
"""
class HeuristicGenerator(object):
    def __init__(self, ):
        print("hello")


loader = DataLoader()
train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_ground, val_ground, test_ground, \
            train_plots, val_plots, test_plots = loader.load_data(None, "./data/imdb/")

model = sklearn.linear_model.LogisticRegression()

synthesizer = Synthesizer(model, train_primitive_matrix, train_ground)
H, X_comb = synthesizer.solve()

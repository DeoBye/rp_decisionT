import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from tqdm import tqdm

import sys
import json
from types import SimpleNamespace
import os
from pathlib import Path


# class EnsembleCfg:
#     def __init__(self, projection_size=10, projection_count=2000, ensemble_size=16, classifier='dt'):
#         self.projection_size = projection_size
#         self.projection_count = projection_count
#         self.ensemble_size = ensemble_size
#         self.classifier = classifier
    
def get_random_projection(data, size):
    P = np.array(range(data.shape[1]))
    np.random.shuffle(P)
    return P[:size]

def train_classifier(X_train, y_train, p, classifier):
    clf=None
    if classifier == 'mnb':
        clf = MultinomialNB()
    if classifier == 'dt':
        clf = DecisionTreeClassifier()
    clf.fit(X_train[:, p], y_train)
    return clf

def pred_classifier(clf, X_test, P):
    y_pred = clf.predict(X_test[:, P])
    return y_pred

def get_classifier(X_train, X_test, y_train, cfg):
    projections = np.array([get_random_projection(X_train, cfg.projection_size) for _ in range(cfg.projection_count)])
    classifiers = [train_classifier(X_train, y_train, P, cfg.classifier) for P in tqdm(projections)]
    predictions = np.array([pred_classifier(clf, X_test, P) for clf, P in tqdm(zip(classifiers, projections))])
    return projections, classifiers, predictions

def max_voting(predictions):
    y_pred = list(map(lambda x: np.bincount(x).argmax(), predictions.transpose()))
    return y_pred

def get_best_ensemble(y_test, projections, predictions, cfg, iter):
    best_accuracy = None
    best_ensemble = None
    count = 0
    logfile_name = Path("./log/10000") / Path(Path(cfg.config).stem + '_10').with_suffix('.json')
    result_name = Path("./result/10000") / Path(Path(cfg.config).stem + '_10').with_suffix('.npy')
    iters = []
    accus = []
    projs = []
    
    for step in tqdm(range(iter)):
        count = count + 1
        ensemble = np.array(range(cfg.projection_count))
        np.random.shuffle(ensemble)
        ensemble = ensemble[:cfg.ensemble_size]
        
        y_pred = max_voting(predictions[ensemble,:])
        this_accuracy = accuracy_score(y_pred, y_test)
        if count > cfg.max_it:
            break
        if best_accuracy is None or this_accuracy > best_accuracy:
            count = 0
            best_ensemble = ensemble
            best_accuracy = this_accuracy
            iters.append(step)
            accus.append(this_accuracy)
            projs.append(projections[ensemble,:].tolist())
            print("Improved accuracy at %d: %f " % (step, this_accuracy))
            np.save("best_projections.npy", projections[ensemble, :])
            # result = {"iter": int(step),
            #           "dt_accuracy": float(this_accuracy), 
            #           "best_projection": str(projections[ensemble, :])} 
    
    with open(logfile_name, 'w') as fp:
        logging = {"projection_size": cfg.projection_size,
                   "ensemble_size": cfg.ensemble_size,
                   "iteration": iters,
                   "accuracy": accus,
                   "projection": projs}
        json.dump(logging, fp)
    np.save(result_name, projections[ensemble, :])
    fp.close()
    


if __name__ == "__main__":
    config=sys.argv[1]
    cfg = json.load(open(config,"r"), object_hook=lambda d: SimpleNamespace(**d))
    cfg.config = os.path.basename(config)
    # ensemble_cfg = EnsembleCfg(cfg.projection_size, cfg.projection_count, cfg.ensemble_size, cfg.classifier)
    
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    print(X_train.shape), print(X_test.shape)
    projections, classifiers, predictions = get_classifier(X_train, X_test, y_train, cfg)
    print(projections.shape)
    print(len(classifiers))
    print(predictions.shape)
    get_best_ensemble(y_test, projections, predictions, cfg, 1000000)
    
    
    
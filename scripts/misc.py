import os
import pickle
import numpy as np
from sklearn import metrics
from sklearn import utils
from sklearn import model_selection
import scipy
import xgboost as xgb
import matplotlib.pyplot as plt


def save_model(model, filename):

    outpath = os.path.join("../Models/", filename)

    with open(outpath, "wb") as f:
        pickle.dump(model, f)

    print("Saved model to file: %s" % (outpath))


def load_model(filename):

    fpath = os.path.join("../Models/", filename)

    with open(fpath, "rb") as f:
        model = pickle.load(f)

    print("Load model to file: %s" % (fpath))
    return model


def regression_results(model, y_true, y_pred):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("-" * 80)
    print("Model: %s" % (model))
    print("-" * 80)
    results = []
    for metric in [metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.mean_absolute_error,
                   metrics.explained_variance_score, metrics.median_absolute_error, metrics.r2_score]:

        res = metric(y_true, y_pred)
        results.append(res)
        print("%s: %.3f" % (metric.__name__, res))
    res = scipy.stats.pearsonr(np.array(y_true),np.array(y_pred))[0]
    results.append(res)
    print("Pearson R: %.3f" %(res))

    print("=" * 80)
    return results


def grid_search_cv(model, parameters, X_train, y_train, n_splits=5, n_iter=1000, n_jobs=-1, scoring="r2", stratified=False):
    """
        Tries all possible values of parameters and returns the best regressor/classifier.
        Cross Validation done is stratified.
        See scoring options at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """

    # Stratified n_splits Folds. Shuffle is not needed as X and Y were already shuffled before.
    if stratified:
        cv = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=42)
    else:
        cv = n_splits

    rev_model = model_selection.RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=cv, scoring=scoring, n_iter=n_iter, n_jobs=n_jobs, random_state=0, verbose=2)
    if (model=="xgb"):
        xgbtrain = xgb.DMatrix(X_train, Y_train)
        output = rev_model.fit(xgbtrain)
        rm(xgbtrain)
        gc()
        return output    
    else:
        return rev_model.fit(X_train, y_train)



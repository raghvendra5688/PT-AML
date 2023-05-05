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


def calculate_regression_metrics(labels, predictions):
    return round(metrics.mean_absolute_error(labels, predictions),3),\
            round(metrics.mean_squared_error(labels, predictions, squared=False),3),\
            round(np.power(scipy.stats.pearsonr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],2),3),\
            round(scipy.stats.pearsonr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],3),\
            round(scipy.stats.spearmanr(np.array(labels).flatten(),np.array(predictions.flatten()))[0],3)


def supervised_learning_steps(method, scoring, data_type, task, model, params, X_train, y_train, n_iter, n_splits = 5):
    
    gs = grid_search_cv(model, params, X_train, y_train, scoring=scoring, n_iter = n_iter, n_splits = n_splits)

    y_pred = gs.predict(X_train)
    y_pred[y_pred < 0] = 0

    if task:
        results=calculate_classification_metrics(y_train, y_pred)
        print("Acc: %.3f, F1: %.3f, AUC: %.3f, AUPR: %.3f" % (results[0], results[1], results[2], results[3]))
    else:
        results=calculate_regression_metrics(y_train,y_pred)
        print("MAE: %.3f, MSE: %.3f, R2: %.3f, Pearson R: %.3f, Spearman R: %.3f" % (results[0], results[1], results[2], results[3], results[4]))
   
    print('Parameters')
    print('----------')
    for p,v in gs.best_estimator_.get_params().items():
        print(p, ":", v)
    print('-' * 80)

    if task:
        save_model(gs, "%s_models/%s_%s_classifier_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_classifier_best_estimator.pk" %(method,method,data_type))
    else:
        save_model(gs, "%s_models/%s_%s_regressor_gs.pk" % (method,method,data_type))
        save_model(gs.best_estimator_, "%s_models/%s_%s_regressor_best_estimator.pk" %(method,method,data_type))
    return(gs)


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



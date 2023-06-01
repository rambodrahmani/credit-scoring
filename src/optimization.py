#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for hyperparamters optimization.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from src import metrics
from src import plotting
from src import utilities
from src import evaluation

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

import optuna

def lr_objective(trial, data, labels, scoring, skf):
    """
    Optuna objective function for Logistic Regression hyperparameters optimization.
    """
    # hyperparameters
    penalty = trial.suggest_categorical('penalty', [None, 'l1', 'l2', 'elasticnet'])
    dual = trial.suggest_categorical('dual', [True, False])
    tol = trial.suggest_float('tol', 0.000001, 1.0)
    C = trial.suggest_float('C', 0.1, 5.0)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr', 'multinomial'])
    warm_start = trial.suggest_categorical('warm_start', [True, False])
    l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)

    # model
    clf = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                             fit_intercept=fit_intercept, class_weight=class_weight,
                             solver=solver, max_iter=max_iter, multi_class=multi_class,
                             warm_start=warm_start, l1_ratio=l1_ratio, n_jobs=-1)

    # evaluate metrics by cross-validation
    scores = cross_validate(clf, data, labels, scoring=scoring, n_jobs=-1, cv=skf)
    scores = np.nan_to_num(scores)
    return scores['test_roc_auc'].mean(), scores['test_emp'].mean()

def optuna_search(objective, model_name, data, target, k_folds,
                  features_scores, features, n_trials, verbose):
    """
    Runs Optuna hyperparameters optimization study using NSGA-II evolutionary
    algorithm.
    """
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    # optimization results save directory
    save_path = evaluation.create_model_save_path(model_name)
    print('Model save_path: ' + save_path)

    # prepare data
    labels = np.array(data[target])
    data = data.drop([target], axis=1, inplace=False)

    # features selection
    if features > 0:
        k_best_features = evaluation.select_k_best_features(features_scores,
                                                            features,
                                                            verbose,
                                                            save_path = save_path)

    # scoring metrics
    scoring = {'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
               'emp': metrics.make_scorer(metrics.emp_score)}

    # stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    study = optuna.create_study(directions=['maximize', 'maximize'],
                                sampler=optuna.samplers.NSGAIISampler())
    study.optimize(lambda trial: objective(trial, data, labels, scoring, skf),
                   n_trials=n_trials, n_jobs=-1, catch=(ValueError, UserWarning),
                   show_progress_bar=True)

    # save study trials csv report
    study.trials_dataframe().to_csv(save_path + 'trials.csv', index=False)
    plotting.plot_pareto_front(study, save_path)

def get_optimal_parameters(trials_path):
    """
    Given the path to the optuna trials.csv output, returns the optimal
    parameters according to the AUC metric.
    """
    ret = ''
    opt_results = utilities.read_csv(trials_path)
    opt_results.sort_values(by=['values_0'], ascending=False, inplace=True)
    for column in opt_results:
        if 'params_' in column:
            param = column.replace('params_', '')
            ret += param + '=' + str(opt_results[column].iloc[0]) + ', '

    return ret
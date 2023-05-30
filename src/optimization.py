#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for hyperparamters optimization.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

from src import utilities
from src import metrics
from src import evaluation

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

import optuna
from optuna.visualization import plot_edf
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_pareto_front
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history

def lr_objective(trial, data, labels, scoring, skf):
    """
    Optuna objective function for Logistic Regression hyperparameters optimization.
    """
    # hyperparameters
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    dual = trial.suggest_categorical('dual', [True, False])
    tol = trial.suggest_float('tol', 1.0e-10, 1.0e-1)
    C = trial.suggest_float('C', 0.1, 5.0)
    fit_intercept = trial.suggest_categorical('fit_intercept', [False, True])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'sag', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr', 'multinomial'])

    # model
    clf = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                             fit_intercept=fit_intercept, solver=solver,
                             max_iter=max_iter, multi_class=multi_class, n_jobs=-1)

    # evaluate metrics by cross-validation
    scores = cross_validate(clf, data, labels, scoring=scoring, n_jobs=-1, cv=skf)
    scores = np.nan_to_num(scores)
    return scores['test_roc_auc'].mean(), scores['test_emp'].mean()

def ct_objective(trial, data, labels, scoring, skf):
    """
    Optuna objective function for Decision Tree hyperparameters optimization.
    """
    # hyperparameters
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    splitter = trial.suggest_categorical('splitter', ['best', 'random'])
    max_depth = trial.suggest_int('max_depth', 1, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 1000)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 1000)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    # model
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                 min_samples_split=min_samples_split, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, class_weight=class_weight)

    # evaluate metrics by cross-validation
    scores = cross_validate(clf, data, labels, scoring=scoring, n_jobs=-1, cv=skf)
    scores = np.nan_to_num(scores)
    return scores['test_roc_auc'].mean(), scores['test_emp'].mean()

def rf_objective(trial, data, labels, scoring, skf):
    """
    Optuna objective function for Random Forest hyperparameters optimization.
    """
    # hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 1, 1000)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_depth = trial.suggest_int('max_depth', 1, 1000)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 1000)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 1000)
    bootstrap = trial.suggest_categorical('bootstrap', [False, True])
    oob_score = trial.suggest_categorical('oob_score', [False, True])

    # model
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
                                 oob_score=oob_score, n_jobs=-1)

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
        k_best_features = evaluation.select_k_best_features(features_scores, features, verbose,
                                                 save_path = save_path + '/' + model_name + '-',)
        print('Selected Features: ' + str(k_best_features))

    # scoring metrics
    scoring = {'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
               'emp': metrics.make_scorer(metrics.emp_score)}

    # stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    study = optuna.create_study(directions=['maximize', 'maximize'],
                                sampler=optuna.samplers.NSGAIISampler())
    study.optimize(lambda trial: objective(trial, data, labels, scoring, skf),
                   n_trials=n_trials, n_jobs=-1, catch=(ValueError, UserWarning), show_progress_bar=True)

    # save study trials
    study.trials_dataframe().to_csv(save_path+'/trials.csv', index=False)

    # plot multi-objective optimization results
    plt = plot_pareto_front(study, target_names=['ROC AUC', 'EMP'])
    plt.write_image(save_path+'/' + model_name + '-' + 'pareto_front.pdf')
    #plt.show()
    plt = plot_optimization_history(study, target=lambda t: t.values[0], target_name="ROC AUC")
    plt.write_image(save_path+'/' + model_name + '-' + 'optimization_history_roc_auc.pdf')
    #plt.show()
    plt = plot_optimization_history(study, target=lambda t: t.values[1], target_name="EMP")
    plt.write_image(save_path+'/' + model_name + '-' + 'optimization_history_emp.pdf')
    #plt.show()
    plt = plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="ROC AUC")
    plt.write_image(save_path+'/' + model_name + '-' + 'parallel_coordinate_roc_auc.pdf')
    #plt.show()
    plt = plot_parallel_coordinate(study, target=lambda t: t.values[1], target_name="EMP")
    plt.write_image(save_path+'/' + model_name + '-' + 'parallel_coordinate_emp.pdf')
    #plt.show()
    #plt = plot_param_importances(study, target=lambda t: t.values[0], target_name="ROC AUC")
    #plt.write_image(save_path+'/' + model_name + '-' + 'param_importances_roc_auc.pdf')
    #plt.show()
    #plt = plot_param_importances(study, target=lambda t: t.values[1], target_name="EMP")
    #plt.write_image(save_path+'/' + model_name + '-' + 'param_importances_emp.pdf')
    #plt.show()

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
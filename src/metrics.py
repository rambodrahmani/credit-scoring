#!/usr/bin/env python

"""
metrics.py: Implementation of utility functions for models metrics evaluation.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from src import plotting

import sklearn
import numpy as np
import pandas as pd
from sklearn import metrics
from EMP.metrics import empCreditScoring

def make_scorer(metric):
    """
    Makes a scorer from a performance metric or loss function.
    """
    return sklearn.metrics.make_scorer(metric)

def roc_auc_score(y_true, y_pred):
    """
    Computes the ROC AUC score using sklearn.metrics.roc_auc_score().
    """
    return metrics.roc_auc_score(y_true, y_pred)

def compute_lgd_point_masses(y_true, y_pred):
    """
    Estimates the bimodal LGD function point masses p0 and p1.
    """
    p_0 = len(np.where(y_true == False)[0])/len(y_true)
    p_1 = len(np.where(y_true == True)[0])/len(y_true)
    return p_0, p_1

def emp_score(y_true, y_pred):
    """
    Estimates the EMP for credit risk scoring, considering constant ROI and a
    bimodal LGD function with point masses p0 and p1 for no loss and total loss,
    respectively. It only returns the score.
    """
    assert (len(y_true) == len(y_pred))
    p_0, p_1 = compute_lgd_point_masses(y_true, y_pred)
    return empCreditScoring(y_pred, y_true, p_0=p_0, p_1=p_1, ROI=0.2644,
                            print_output=False)[0]

def emp_score_frac(y_true, y_pred):
    """
    Estimates the EMP for credit risk scoring, considering constant ROI and a
    bimodal LGD function with point masses p0 and p1 for no loss and total loss,
    respectively. It returns both the score and the fraction of excluded.
    """
    assert (len(y_true) == len(y_pred))
    p_0, p_1 = compute_lgd_point_masses(y_true, y_pred)
    return empCreditScoring(y_pred, y_true, p_0=p_0, p_1=p_1, ROI=0.2644,
                            print_output=False)
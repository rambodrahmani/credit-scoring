#!/usr/bin/env python

"""
evaluation.py: Implementation of utility functions for evaluating models.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import numpy as np
from tqdm import tqdm

from src import plotting
from src import utilities

from datetime import datetime

def select_k_best_features(features_scores, k, verbose=True, save_path=None):
    """
    Selects the k best features based on the scores obtained using different
    features slection techniques.
    """
    features_scores = sorted(features_scores.items(), key=lambda x: x[1], reverse=True)
    features_scores = dict(features_scores)
    k_best_features = list(features_scores.keys())[:k]

    if verbose:
        plotting.plot_features_scores(features_scores, '',
                                      figsize=(12, 3),
                                      save_path=save_path, dpi=100)
        print('Selected Features: ' + str(k_best_features))

    return k_best_features

def create_model_save_path(model_name):
    """
    Creates train and test results save directory for the given
    model name.
    """
    sub_directories = model_name.split('-')

    ret = '/'.join(utilities.__file__.split('/')[:-2]) + '/models'
    utilities.create_directory(ret)

    for sub_directory in sub_directories:
        ret = os.path.join(ret, sub_directory)
        utilities.create_directory(ret)

    ret = os.path.join(ret, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.isdir(ret):
        os.mkdir(ret)

    return ret + '/'
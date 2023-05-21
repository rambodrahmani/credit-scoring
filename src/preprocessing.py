#!/usr/bin/env python

"""
preprocessing.py: Implementation of utility functions for preprocessing.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import pandas as pd
from tqdm import tqdm
from optbinning import OptimalBinning

def optbinning_woe(data, solver, outlier_detector, save_path, verbose):
    """
    Applies WoE encoding after binning using OptimalBinning to the given
    dataframe features and computes IV scores.

    :param data: pandas.DataFrame to be preprocessed.
    :param solver: OptimalBinning solver to be used (cp, mip).
    :param outlier_detector: OptimalBinning outlier_detector to be used (range, zscore).
    :param verbose: set to True for verbose logging.

    :return: preprocessed pandas.DataFrame and features scores dictionary.
    """ 
    features_scores = {}
    column_dtype = ''

    optbinning_save_path = os.path.join(save_path, 'optbinning')
    if not os.path.isdir(optbinning_save_path):
        os.mkdir(optbinning_save_path)

    for column in tqdm(data.dtypes.index):
        if verbose:
            print('Processing feature: ' + column + '.')

        if data.dtypes[column] in ['float64', 'int64']:
            column_dtype = "numerical"
        elif data.dtypes[column] in ['category']:
            column_dtype = "categorical"
        elif data.dtypes[column] in ['bool']:
            continue

        x = data[column].values
        y = data.defaulted
        optb = OptimalBinning(name=column, dtype=column_dtype, solver=solver,
                              outlier_detector=outlier_detector, verbose=verbose)
        optb.fit(x, y)
        binning_table = optb.binning_table
        binning_table_df = binning_table.build()
        features_scores[column] = binning_table_df['IV']['Totals']
        binning_table.plot(metric="woe", show_bin_labels=True,
                           savefig=optbinning_save_path+'/'+column.replace('/', '-')+'-binning.pdf')
        data[column] = optb.transform(x, metric="woe")
        
        if verbose:
            print('Solver status for feature ' + column + ': ' + optb.status)
            print(binning_table_df['Bin'])
            print(pd.Series(data[column]).value_counts())

    return data, features_scores

def features_correlation(data):
    """
    Computes features correlation using both Pearson and Spearman coefficients.
    """
    pearson_cor = data.corr(method='pearson')
    spearman_cor = data.corr(method='spearman')
    cor = (pearson_cor+spearman_cor)/2

    return cor
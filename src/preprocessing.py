#!/usr/bin/env python

"""
preprocessing.py: Implementation of utility functions for preprocessing.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

from src import utilities

import os
import json
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.model_selection import train_test_split

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
    utilities.create_directory(optbinning_save_path)

    for column in data.dtypes.index:
        if verbose:
            print('\n\nProcessing feature: ' + column + '.')

        if data.dtypes[column] in ['float64', 'int64']:
            column_dtype = "numerical"
        elif data.dtypes[column] in ['category']:
            column_dtype = "categorical"
        elif data.dtypes[column] in ['bool']:
            continue

        x = data[column].values
        y = data.defaulted
        optb = OptimalBinning(name=column, dtype=column_dtype, solver=solver,
                              outlier_detector=outlier_detector, time_limit=1000,
                              verbose=verbose)
        optb.fit(x, y)
        binning_table = optb.binning_table
        binning_table_df = binning_table.build()
        features_scores[column] = binning_table_df['IV']['Totals']
        binning_table.plot(metric="woe", show_bin_labels=True,
                           savefig=optbinning_save_path+'/'+column.replace('/', '-')+'-binning.pdf')
        data[column] = optb.transform(x, metric="woe")
        
        if verbose:
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

def compute_p1(data):
    """
    Computes the p1 mass point of the LGD (Loss Given Default) function.
    """
    return len(data[data['defaulted'] == True])/len(data)

def compute_p0(data):
    """
    Computes the p0 mass point of the LGD (Loss Given Default) function.
    """
    return len(data[data['defaulted'] == False])/len(data)

def get_to_be_dropped(f_corr, features_scores, corr_thr, verbose):
    """
    Finds correlated features to be dropped based on the IV score.
    """
    to_be_dropped = []
    
    for index, row in f_corr.iterrows():
        for feat_name, correlation_score in row.items():
            if feat_name != 'defaulted' and correlation_score > corr_thr:
                if features_scores[feat_name] > features_scores[index]:
                    if verbose:
                        print(f"{index} IS CORRELATED WITH {feat_name}: {correlation_score}")
                        print(f"THE IV of {index} is {features_scores[index]}")
                        print(f"THE IV of {feat_name} is {features_scores[feat_name]}")
                        print(f"DROP {index}\n")
                    to_be_dropped.append(index)
                    break
    
    return list(np.unique(to_be_dropped))

def save_dataset(data, features_scores, test_size, save_path):
    """
    Splits the preprocessed dataset in train and test and saves the dataframes
    using the .parquet file format. The features scores dictionary is saved as well.
    """
    # store features selection scores
    with open(os.path.join(save_path, 'features_scores.json'), 'w') as outfile:
        json.dump(features_scores, outfile, indent=4)

    train_split, test_split = train_test_split(data, test_size=test_size, shuffle=True, stratify=data[['defaulted']])
    train_split.to_parquet(os.path.join(save_path, 'train.parquet'), index=False)
    test_split.to_parquet(os.path.join(save_path, 'test.parquet'), index=False)

    print("Train split size:", len(train_split))
    print("Train split p1:", compute_p1(train_split))
    print("Train split p0:", compute_p0(train_split))

    print("\nTest split size:", len(test_split))
    print("Test split p1:", compute_p1(test_split))
    print("Test split p0:", compute_p0(test_split))
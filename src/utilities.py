#!/usr/bin/env python

"""
utilities.py: Implementation of general purpose utility functions.
"""

__author__      = "Rambod Rahmani <rambodrahmani@autistici.org>"
__copyright__   = "Rambod Rahmani 2023"

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# show all pandas dataframe columns
pd.set_option('display.max_columns', None)

def create_directory(dir_path):
    """
    Creates the specified directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def read_csv(dataset_path, dtype=None, sep=',', header='infer'):
    """
    Loads the provided .csv file as a pandas Dataframe.
    """
    return pd.read_csv(dataset_path, dtype=dtype, sep=sep, header=header)

def read_excel(dataset_path, dtype=None, header=0):
    """
    Loads the provided .xls file as a pandas Dataframe
    """
    return pd.read_excel(dataset_path, dtype=dtype, header=header)

def read_parquet(dataset_path):
    """
    Loads the provided .parquet file as a pandas Dataframe
    """
    return pd.read_parquet(dataset_path)

def int64_to_float64(data):
    """
    Converts all int64 columns of the dataframe to float64.
    """
    int_columns = data.select_dtypes(include='int64').columns
    data[int_columns] = data[int_columns].astype('float64')
    return data

def object_to_category(data):
    """
    Converts all object columns of the dataframe to category.
    """
    int_columns = data.select_dtypes(include='object').columns
    data[int_columns] = data[int_columns].astype('category')
    return data

def object_to_float64(data):
    """
    Converts all object columns of the dataframe to float64.
    """
    int_columns = data.select_dtypes(include='object').columns
    data[int_columns] = data[int_columns].astype('float64')
    return data

def replace_to_nan(data, word):
    """
    Replaces the given word with np.nan.
    """
    data.replace(word, np.nan, inplace=True)

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
    print("Test split size:", len(test_split))

def read_features_scores(dataset_path):
    """
    Reads and returns the features_scores.json file for the given dataset.
    """
    file_path = os.path.join(dataset_path, 'features_scores.json')
    with open(file_path) as json_file:
        ret = json.load(json_file)

    return ret

def log(x):
    """
    Computes the natural logarithm, element-wise.
    """
    return np.log(x)

def print_unique_values(data):
    """
    Prints the unique values for each column in the provided DataFrame.
    """
    for column in data.columns:
        unique_values = data[column].unique().tolist()
        print(f'Unique values for {column}: {unique_values}')
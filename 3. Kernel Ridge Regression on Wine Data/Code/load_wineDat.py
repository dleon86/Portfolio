# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:45:11 2022

@author: danny
"""

import numpy as np
import pandas as pd

def load_wineDat():
    
    trainDat_df = pd.read_csv('wine_training.csv')
    """
    5. Attributes information:
    
       Input features (columns 1-11 in CSV files):
    
       1 - fixed acidity
       2 - volatile acidity
       3 - citric acid
       4 - residual sugar
       5 - chlorides
       6 - free sulfur dioxide
       7 - total sulfur dioxide
       8 - density
       9 - pH
       10 - sulphates
       11 - alcohol
    
       Output (column 12 in CSV file): 
       12 - quality (score between 0 and 10)
    """
    # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    #                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
    #                    'density', 'pH', 'sulphates', 'alcohol']
    cols = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
                       'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 
                       'Density', 'pH', 'Sulphates', 'Alcohol']
    # #%%
    X_train_df = trainDat_df.iloc[:,:-1]
    Y_train_df = trainDat_df.iloc[:,-1]
    
    X_train_df.columns = cols
    
    Y_train_df.columns = ['quality']
    
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.colheader_justify', 'center')
    # pd.set_option('display.precision', 6)
    
    # display(X_train_df.head())
    
    X_train = np.array(X_train_df)
    Y_train = np.array(Y_train_df)
    
    # Next we normalize and center the training set
    
    X_train_N = X_train.shape[0]
    
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    
    X_train_normal = (X_train - np.matlib.repmat(X_train_mean, X_train_N, 1))/np.matlib.repmat(X_train_std, X_train_N, 1)
    
    Y_train_N = Y_train.shape[0]
    
    Y_train_mean = np.mean(Y_train, axis=0)
    Y_train_std = np.std(Y_train, axis=0)
    
    Y_train_normal = (Y_train - Y_train_mean)/Y_train_std
    
    Y_dat = Y_train_std, Y_train_mean, X_train_std, X_train_mean
    
    print(X_train_normal.shape)
    print(Y_train_normal.shape)
    # #%%
    testDat_df = pd.read_csv('wine_test.csv')
    X_test_df = testDat_df.iloc[:,:-1]
    Y_test_df = testDat_df.iloc[:,-1]
    
    X_test_df.columns = cols
    
    Y_test_df.columns = ['quality']
    
    # print(df.to_latex(index=False)) # print LaTeX of dataframe
    
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.colheader_justify', 'center')
    # pd.set_option('display.precision', 6)
    # display(X_test_df.head())
    # print(df.to_latex(index=False)) # print LaTeX of dataframe
    
    X_test = np.array(X_test_df)
    Y_test = np.array(Y_test_df)
    
    # X_test_mean = np.mean(X_test, axis=0)
    # X_test_std = np.std(X_test, axis=0)
    
    # Y_test_mean = np.mean(Y_test, axis=0)
    # Y_test_std = np.std(Y_test, axis=0)
    
    X_test_N = X_test.shape[0]
    
    X_test_normal = (X_test - np.matlib.repmat(X_train_mean, X_test_N, 1))/np.matlib.repmat(X_train_std, X_test_N, 1)
    Y_test_normal = (Y_test - Y_train_mean)/Y_train_std
    # X_test_normal = (X_test - np.matlib.repmat(X_test_mean, X_test_N, 1))/np.matlib.repmat(X_test_std, X_test_N, 1)
    # Y_test_normal = (Y_test - Y_test_mean)/Y_test_std
    
    
    X_new_df = pd.read_csv('wine_new_batch.csv', header=None)
    
    X_new = np.array(X_new_df)
    X_new_N = X_new.shape[0]
    X_new = (X_new - np.matlib.repmat(X_train_mean, X_new_N, 1))/np.matlib.repmat(X_train_std, X_new_N, 1)
    
    return X_train, Y_train, X_train_normal, Y_train_normal, X_test, Y_test, X_test_normal, Y_test_normal, X_train_df, Y_dat, X_new

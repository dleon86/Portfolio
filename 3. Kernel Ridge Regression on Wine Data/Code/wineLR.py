# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:38:21 2022

@author: danny
"""
# import numpy as np
# import numpy.matlib
# import matplotlib
import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import 
from sklearn.metrics import mean_squared_error
import sklearn as skl
import sklearn.model_selection
import sklearn.kernel_ridge

from plot_Wine2 import plot_Wine2


def wineLR(X_train_nrm, Y_train_nrm, X_test_nrm, Y_test_nrm, X_train_df, Y_dat):
# #%% Linear regression
 
    reg = LinearRegression().fit(X_train_nrm, Y_train_nrm)
    
    # predict the training data
    Y_train_nrm_P = reg.predict(X_train_nrm)
    
    # use our fitted model to predict the Y values on the test set 
    Y_test_nrm_P = reg.predict(X_test_nrm)
    
    this_score = skl.model_selection.cross_val_score(reg, X_train_nrm, Y_train_nrm)
    
    # plot the predicted values of Y against the test set
    
    MSEtrain = mean_squared_error( Y_train_nrm, Y_train_nrm_P)
    MSEtest = mean_squared_error( Y_test_nrm, Y_test_nrm_P)
    
    m,n = 4,3
    
    dtype = 'Linear Regression'
    banner = f'Predictions on Training Wine Data\n {dtype}\n'
    
    Y_train = Y_train_nrm*Y_dat[1]+Y_dat[0]
    Y_test = Y_test_nrm*Y_dat[1]+Y_dat[0]
    Y_test_P = Y_test_nrm_P*Y_dat[1]+Y_dat[0]
    
    plot_Wine2(X_train_nrm, Y_train, X_test_nrm, Y_test, X_train_df, Y_test_P, banner)
    
    print('Linear Regression Score: ', this_score, '\n MSEtrain: ' , MSEtrain, '\n MSEtest: ' , MSEtest)

    return this_score, MSEtrain, MSEtest
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:19:38 2022

@author: danny
"""

from sklearn.metrics import mean_squared_error
# import sklearn as skl
# import sklearn.model_selection
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, cross_val_score

from plot_Wine2 import plot_Wine2

def wineKR(kern, sigma, alph, X_train_nrm, Y_train_nrm, X_test_nrm, Y_test_nrm, X_train_df, banner, Y_dat):
    
    KRR = KernelRidge(kernel=kern, alpha = alph, gamma=1/(2*sigma**2)) #'rbf','laplacian'
    
    KRR.fit(X_train_nrm, Y_train_nrm)
    
    # use our fitted model to predict the Y values on the test set 
    Y_test_nrm_P = KRR.predict(X_test_nrm)
    
    # predict the training data
    Y_train_nrm_P = KRR.predict(X_train_nrm)
    
    this_score = cross_val_score(KRR, X_train_nrm, Y_train_nrm)
    
    # plot the predicted values of Y against the test set
    
    MSEtrain = mean_squared_error( Y_train_nrm, Y_train_nrm_P)
    MSEtest = mean_squared_error( Y_test_nrm, Y_test_nrm_P)
    
    Y_train = Y_train_nrm*Y_dat[1]+Y_dat[0]
    Y_test = Y_test_nrm*Y_dat[1]+Y_dat[0]
    Y_test_P = Y_test_nrm_P*Y_dat[1]+Y_dat[0]
    
    plot_Wine2(X_train_nrm, Y_train, X_test_nrm, Y_test, X_train_df, Y_test_P, banner)
    
    return this_score, MSEtrain, MSEtest
  
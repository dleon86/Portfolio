# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:00:02 2022

@author: Daniel Leon
"""
# # clear all variables
# from IPython import get_ipython
# get_ipython().magic('reset -sf')


import time
import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error
import sklearn as skl
from sklearn.kernel_ridge import KernelRidge

from load_wineDat import load_wineDat

X_train, Y_train, X_train_nrm, Y_train_nrm, X_test, Y_test, X_test_nrm, Y_test_nrm, X_train_df, XY_dat, X_new = load_wineDat()


# #%%
# range of values of sigma to try 

t0 = time.time()

K_sgm = 10, 10
K_lmbd = 10, 10


sgm1 = np.linspace(2, 6, K_sgm[0])
lmbd1 = np.linspace(-5, 0, K_lmbd[0])


sgm2 = np.linspace(3, 7, K_sgm[1])
lmbd2 = np.linspace(-5, 0, K_lmbd[1])

scores = np.zeros((K_sgm[0], K_lmbd[0])), np.zeros((K_sgm[1], K_lmbd[1]))
scores_std = np.zeros((K_sgm[0], K_lmbd[0])), np.zeros((K_sgm[1], K_lmbd[1]))

MSEtrain = np.zeros((K_sgm[0], K_lmbd[0])), np.zeros((K_sgm[1], K_lmbd[1]))
MSEtest = np.zeros((K_sgm[0], K_lmbd[0])), np.zeros((K_sgm[1], K_lmbd[1]))

# #%%

rbf_KRR_CV = KernelRidge(kernel='rbf')
lap_KRR_CV = skl.kernel_ridge.KernelRidge(kernel='laplacian')

for i in range(K_sgm[0]):

    rbf_KRR_CV.gamma = 1/(2*sgm1[i]**2)    
    lap_KRR_CV.gamma = 1/(sgm2[i])
    
    for j in range((K_lmbd[0])):

        rbf_KRR_CV.alpha = (2**lmbd1[j])
        lap_KRR_CV.alpha = (2**lmbd2[j])
        
        # t0 = time.time()
        rbf_KRR_CV.fit(X_train_nrm, Y_train_nrm)
        lap_KRR_CV.fit(X_train_nrm, Y_train_nrm)

        Y_train_nrm_rbfP = rbf_KRR_CV.predict(X_train_nrm)
        Y_train_unrm_rbfP = (Y_train_nrm_rbfP*XY_dat[0] + XY_dat[1]).round()
        Y_train_nrm_lapP = lap_KRR_CV.predict(X_train_nrm) 
        Y_train_unrm_lapP = (Y_train_nrm_lapP*XY_dat[0] + XY_dat[1]).round()
        
        Y_test_nrm_rbfP = rbf_KRR_CV.predict(X_test_nrm)
        Y_test_unrm_rbfP = (Y_test_nrm_rbfP*XY_dat[0] + XY_dat[1]).round()
        Y_test_nrm_lapP = lap_KRR_CV.predict(X_test_nrm)
        Y_test_unrm_lapP = (Y_test_nrm_lapP*XY_dat[0] + XY_dat[1]).round()

        MSEtrain[0][i,j] = mean_squared_error(Y_train, Y_train_unrm_rbfP)
        MSEtrain[1][i,j] = mean_squared_error(Y_train, Y_train_unrm_lapP)
        
        MSEtest[0][i,j] = mean_squared_error(Y_test, Y_test_unrm_rbfP)
        MSEtest[1][i,j] = mean_squared_error(Y_test, Y_test_unrm_lapP)



print('\n MSEtrain_rbf',MSEtrain[0],'\n MSEtrain_lap',MSEtrain[1], \
      '\n MSEtest_rbf', MSEtest[0], '\n MSEtest_lap', MSEtest[1])

print('Time fitting data: ', time.time() - t0)    
    
# #%%
S1, L1 = np.meshgrid(sgm1, lmbd1)
S2, L2 = np.meshgrid(sgm2, lmbd2)

fig = make_subplots(
    rows=1, cols=2, subplot_titles=(
        "Gaussian Kernel", 
        "Laplacian Kernel"),
        
        specs=[[{'type': 'surface'}, {'type': 'surface'}]])


# Add traces
fig.add_trace(go.Surface( x = L1, y = S1, z =  np.log2(MSEtrain[0]),
                         colorscale='Viridis', showscale=True, 
                         colorbar=dict(x=.45, title='train', tickfont=dict(size=18))),
              row=1, col=1
              )
fig.add_trace(go.Surface( x = L1, y = S1, z =  np.log2(MSEtest[0]),
                          colorscale='RdBu', showscale=True, 
                          colorbar=dict(x=.5, title='test', tickfont=dict(size=18))),
              row=1, col=1
              )
fig.add_trace(go.Surface( x = L2, y = S2, z =  np.log2(MSEtrain[1]),
                         colorscale='Viridis', showscale=True, 
                         colorbar=dict(x=1, title='train', tickfont=dict(size=18))),
              row=1, col=2
              )
fig.add_trace(go.Surface( x = L2, y = S2, z =  np.log2(MSEtest[1]),
                          colorscale='RdBu', showscale=True, 
                          colorbar=dict(x=1.05, title='test', tickfont=dict(size=18))),
              row=1, col=2
              )

fig.update_layout(
    title_text='Hyperparameter Tuning - MSE', title_x=0.5, autosize=False, 
    height = 900, width=1900,
    margin=dict(r=40, l=40, t=80, b=80),
    scene = dict(
                    xaxis = dict(
                        title='λ'),
                    yaxis = dict(
                        title='σ'),#σ²Gaussian-γ
                    zaxis = dict(
                        title='log₂(MSE)'),),
    scene2 = dict(
                    xaxis = dict(
                        title='λ'),
                    yaxis = dict(
                        title='σ'),#σ²Laplacian-γ
                    zaxis = dict(
                        title='log₂(MSE)'),),)
 
fig.layout.title.font.size=42 
fig.layout.annotations[0].font.size=32
fig.layout.annotations[1].font.size=32

fig.layout.scene.xaxis.tickfont.size=18
fig.layout.scene.yaxis.tickfont.size=18
fig.layout.scene.zaxis.tickfont.size=18
fig.layout.scene2.xaxis.tickfont.size=18
fig.layout.scene2.yaxis.tickfont.size=18
fig.layout.scene2.zaxis.tickfont.size=18

fig.layout.scene.xaxis.titlefont.size=30
fig.layout.scene2.xaxis.titlefont.size=30
fig.layout.scene.yaxis.titlefont.size=30
fig.layout.scene2.yaxis.titlefont.size=30
fig.layout.scene.zaxis.titlefont.size=30
fig.layout.scene2.zaxis.titlefont.size=30
    
fig.layout.scene.yaxis.nticks=5
fig.layout.scene2.yaxis.nticks=5

fig.layout.scene.zaxis.nticks=3
fig.layout.scene2.zaxis.nticks=5

fig.show(renderer="browser")


# #%%
# Locate minimum MSE
rbftrain_min = np.where(MSEtrain[0]==MSEtrain[0].min())
rbftrain_min = rbftrain_min[0][0], rbftrain_min[1][0]

rbftest_min = np.where(MSEtest[0]==MSEtest[0].min())
rbftest_min = rbftest_min[0][0], rbftest_min[1][0]

laptrain_min = np.where(MSEtrain[1]==MSEtrain[1].min())
laptrain_min = laptrain_min[0][0], laptrain_min[1][0]

laptest_min = np.where(MSEtest[1]==MSEtest[1].min())
laptest_min = laptest_min[0][0], laptest_min[1][0]

print('Gaussian Kernel MSEtrain:', MSEtrain[0][rbftest_min[0],rbftest_min[1]],\
      '\n with minimized MSEtest:', MSEtest[0][rbftest_min[0],rbftest_min[1]],\
      '\n at σ and λ: (', sgm1[rbftest_min[0]], lmbd1[rbftest_min[1]], \
          ')\n laplacian Kernel MSEtrain:',  MSEtrain[0][laptest_min[0],laptest_min[1]],\
               '\n with minimized MSEtest:',  MSEtest[0][laptest_min[0],laptest_min[1]],\
          '\n at σ and λ: (',sgm2[laptest_min[0]], lmbd2[laptest_min[1]],')')

#%%
#  Kernel Ridge, gaussian

m,n = 4,3

# see https://scikit-learn.org/stable/modules/metrics.html#metrics for possible choices of kernels
sigma = 3.0847# sgm1[rbftest_min[0]]

alph = 1.2723# 2**lmbd1[rbftest_min[1]]

KRR = skl.kernel_ridge.KernelRidge(kernel='rbf', alpha = alph, gamma=1/(2*sigma**2)) #'rbf','laplacian'

KRR.fit(X_train_nrm, Y_train_nrm)

# use our fitted model to predict the Y values on the test set 
Y_test_nrm_P = KRR.predict(X_test_nrm)
Y_test_unrm_P = (Y_test_nrm_P*XY_dat[0] + XY_dat[1]).round()

# predict the training data
Y_train_nrm_P = KRR.predict(X_train_nrm)
Y_train_unrm_P = (Y_train_nrm_P*XY_dat[0] + XY_dat[1]).round()

Y_new_nrm_P = KRR.predict(X_new)
Y_new_unrm_P = (Y_new_nrm_P*XY_dat[0] + XY_dat[1]).round()

this_score_rbf = skl.model_selection.cross_val_score(KRR, X_train_nrm, Y_train_nrm)
        
# plot the predicted values of Y against the test set

MSEtrain_rbf = mean_squared_error( Y_train, Y_train_unrm_P)
MSEtest_rbf = mean_squared_error( Y_test, Y_test_unrm_P)

X_new_unrm = X_new*XY_dat[2] + XY_dat[3]

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Optimized Gaussian Kernel Predictions on Training Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_rbf: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_rbf: .4f} \
             \n $\\sigma =$ {sigma: .4f}, $\\lambda =$ {alph: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off')
        break
    else:
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train, color='lime', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train_unrm_P, color='darkviolet', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='gold', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)

fig.show()

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Optimized Gaussian Kernel Predictions on Test Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_rbf: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_rbf: .4f} \
             \n $\\sigma =$ {sigma: .4f}, $\\lambda =$ {alph: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off')
        break
    else:
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test, color='mediumspringgreen', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test_unrm_P, color='blueviolet', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='gold', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)

fig.show()

print('Kernel Ridge Score: ', this_score_rbf, '\n MSEtrain: ' , MSEtrain_rbf, '\n MSEtest: ' , MSEtest_rbf)

print('Score on New Batch:', Y_new_unrm_P, Y_new_unrm_P.mean())


#%%
#  Kernel Ridge, Laplacian

m,n = 4,3

# see https://scikit-learn.org/stable/modules/metrics.html#metrics for possible choices of kernels
sigma =  sgm2[laptest_min[0]]#0.0448#

alph =  2**lmbd2[laptest_min[1]]#1.2387#

KRR = skl.kernel_ridge.KernelRidge(kernel='laplacian', alpha = alph, gamma=1/(sigma)) #'rbf','laplacian'

KRR.fit(X_train_nrm, Y_train_nrm)


# use our fitted model to predict the Y values on the test set 
Y_test_nrm_P = KRR.predict(X_test_nrm)
Y_test_unrm_P = (Y_test_nrm_P*XY_dat[0] + XY_dat[1]).round()

# predict the training data
Y_train_nrm_P = KRR.predict(X_train_nrm)
Y_train_unrm_P = (Y_train_nrm_P*XY_dat[0] + XY_dat[1]).round()

Y_new_nrm_P = KRR.predict(X_new)
Y_new_unrm_P = (Y_new_nrm_P*XY_dat[0] + XY_dat[1]).round()

this_score_lap = skl.model_selection.cross_val_score(KRR, X_train_nrm, Y_train_nrm)
        
# plot the predicted values of Y against the test set

MSEtrain_lap = mean_squared_error( Y_train, Y_train_unrm_P)
MSEtest_lap = mean_squared_error( Y_test, Y_test_unrm_P)

X_new_unrm = X_new*XY_dat[2] + XY_dat[3]


# print(Y_new_unrm_P)

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Optimized Laplacian Kernel Predictions on Training Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_lap: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_lap: .4f}, \
                 \n $\sigma =$ {sigma: .4f}, $\\lambda =$ {alph: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off')
        break
    else:
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train, color='lime', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train_unrm_P, color='blueviolet', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='yellow', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)
        
fig.show()

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Optimized Laplacian Kernel Predictions on Test Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_lap: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_lap: .4f}, \
                 \n $\sigma =$ {sigma: .4f}, $\\lambda =$ {alph: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off')
        break
    else:
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test, color='cyan', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test_unrm_P, color='darkviolet', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='yellow', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)
        
fig.show()



print('LAplacian Kernel Ridge Score: ', this_score_lap, '\n MSEtrain: ' , MSEtrain_lap, '\n MSEtest: ' , MSEtest_lap)
print('Score on New Batch:', Y_new_unrm_P, Y_new_unrm_P.mean())

#%%
#  Kernel Ridge, Laplacian

m,n = 4,3

KRR = skl.kernel_ridge.KernelRidge(kernel='linear')

KRR.fit(X_train_nrm, Y_train_nrm)


# use our fitted model to predict the Y values on the test set 
Y_test_nrm_P = KRR.predict(X_test_nrm)
Y_test_unrm_P = (Y_test_nrm_P*XY_dat[0] + XY_dat[1]).round()

# predict the training data
Y_train_nrm_P = KRR.predict(X_train_nrm)
Y_train_unrm_P = (Y_train_nrm_P*XY_dat[0] + XY_dat[1]).round()

Y_new_nrm_P = KRR.predict(X_new)
Y_new_unrm_P = (Y_new_nrm_P*XY_dat[0] + XY_dat[1]).round()

this_score_lr = skl.model_selection.cross_val_score(KRR, X_train_nrm, Y_train_nrm)
        
# plot the predicted values of Y against the test set

MSEtrain_lr = mean_squared_error( Y_train, Y_train_unrm_P)
MSEtest_lr = mean_squared_error( Y_test, Y_test_unrm_P)

X_new_unrm = X_new*XY_dat[2] + XY_dat[3]


print(Y_new_unrm_P)

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Linear Regression Predictions on Training Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_lr: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_lr: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off') 
        break
    else:
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train, color='springgreen', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_train[:, i+ j*n], Y_train_unrm_P, color='coral', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='gold', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)

fig.show()

fig, ax = plt.subplots(m,n, figsize=(24,32))

fig.suptitle(f'Linear Regression Predictions on Test Wine Data \
             \n ${{\\bf MSE}}_{{train}} =${MSEtrain_lr: .4f}, ${{\\bf MSE}}_{{test}} =${MSEtest_lr: .4f}\n',
             fontsize=32, fontweight='bold', horizontalalignment='center', x=.55)


for j in range(m):
  for i in range(n):

    if ((j==m-1) & (i==n-1)):
        ax[j][i].axis('off')
        break
    else:
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test, color='chartreuse', 
                          label='Test', s=256, edgecolors='k', linewidth=1)
        ax[j][i].scatter( X_test[:, i+ j*n], Y_test_unrm_P, color='salmon', 
                          label='Prediction', s=256, edgecolors='k', linewidth=1, 
                          marker='x')
        ax[j][i].scatter( X_new_unrm[:, i+ j*n], Y_new_unrm_P, color='gold', 
                          label='New Batch', s=256, edgecolors='k', linewidth=1 )
        fig.tight_layout()
        
        ax[j][i].legend(fontsize=22)
        ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=28, fontweight='bold')
        # Adjust the x,y-tick font size
        for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
            label.set_fontsize(22)
            
    if (i==0):
        ax[j][i].set_ylabel('Wine Score\n', fontsize=32, fontweight='bold')
    else:
        ax[j][i].axes.get_yaxis().set_visible(False)

print('LR new batch Score: ', Y_new_unrm_P, '\n MSEtrain: ' , MSEtrain_lr, '\n MSEtest: ' , MSEtest_lr)

# #%%
# import pandas as pd

# b = 4
# x = pd.DataFrame({"col" : [b]})

# t = time.localtime()
# timestamp = time.strftime('%b-%d-%Y_%H%M', t)

# writer = pd.ExcelWriter('\\MSEtrain_' + timestamp  + ".xlsx")

# x.to_excel(writer,sheet_name="MSEtrain", index = False)

# writer.save()
# print("done")


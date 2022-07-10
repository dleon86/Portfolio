# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:21:08 2022

@author: danny
"""
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_Wine(X_train, Y_train, X_test, Y_test, X_train_df, dtype):
    
    # #%% Plot the data raw
    # set colormap
    cmap = matplotlib.cm.get_cmap('gist_stern')#'gist_ncar','nipy_spectral','turbo','tab20c','jet','gist_rainbow'
    cmap2 = matplotlib.cm.get_cmap('gist_rainbow')#'gist_ncar','gist_stern'
    
    m,n = 4,3
    fig, ax = plt.subplots(m,n, figsize=(24,18))
    # cmap = 'tab20'
    
    fig.suptitle(f'{dtype} Wine Data', fontsize=32, fontweight='bold')
    
    for j in range(m):
      for i in range(n):
    
        if ((j==m-1) & (i==n-1)):
            ax[j][i].axis('off')
            break
        else:
            ax[j][i].scatter( X_train[:, i+ j*n], Y_train,color=cmap(((i+ j*n)*1/11)), s=128, 
                             edgecolors='k', linewidth=1.5)
            ax[j][i].scatter( X_test[:, i+ j*n], Y_test,color=cmap2(((i+ j*n)*1/11)), s=128, 
                             edgecolors='k', linewidth=1.5)
    
            # ax[j][i].set_xlabel('x_'+str(i + j*n), fontsize=20)
            fig.tight_layout()
            ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=22, fontweight='bold')
            # Adjust the x,y-tick font size
            ax[j][i].legend(('Training Data', 'Test Data'), fontsize=20)
            for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
                label.set_fontsize(20)
            
            # fig.tight_layout()
            
        if (i==0):
            ax[j][i].set_ylabel('Wine Score', fontsize=24, fontweight='bold')
        else:
            ax[j][i].axes.get_yaxis().set_visible(False)
        
# fig.show()

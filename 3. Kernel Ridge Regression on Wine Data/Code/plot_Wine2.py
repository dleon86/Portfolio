# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:46:02 2022

@author: danny
"""

import matplotlib.pyplot as plt
# import matplotlib


def plot_Wine2(X_train_nrm, Y_train_nrm, X_test_nrm, Y_test_nrm, X_train_df, Y_pred_nrm, banner):

    
    m,n = 4,3
    fig, ax = plt.subplots(m,n, figsize=(24,18))
    # cmap = 'tab20'
    fig.suptitle(banner, fontsize=32, fontweight='bold')
    
    
    for j in range(m):
      for i in range(n):
    
        if ((j==m-1) & (i==n-1)):
            ax[j][i].axis('off')
            break
        else:
            ax[j][i].scatter( X_test_nrm[:, i+ j*n], Y_test_nrm, color='c', 
                             label='Test', s=128, edgecolors='k', linewidth=1)
            ax[j][i].scatter( X_test_nrm[:, i+ j*n], Y_pred_nrm, color='m', 
                             label='Prediction', s=128, edgecolors='k', linewidth=1 )
            fig.tight_layout()
            
            ax[j][i].legend(('test', 'prediction'), fontsize=20)
            ax[j][i].set_xlabel(X_train_df.columns[(i + j*n)], fontsize=22, fontweight='bold')
            # Adjust the x,y-tick font size
            for label in (ax[j][i].get_xticklabels() + ax[j][i].get_yticklabels()):
                label.set_fontsize(20)
                
        if (i==0):
            ax[j][i].set_ylabel('Wine Score\n', fontsize=24, fontweight='bold')
        else:
            ax[j][i].axes.get_yaxis().set_visible(False)
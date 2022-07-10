import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("./input"))

#%% Assignment
dfg = pd.read_csv('./input/CC GENERAL.csv')
# dfg.head()

# look at describe in data viewer
# dfg.describe()

# # Let's see our data in a detailed way with pairplot
# sns.pairplot(dfg, hue='BALANCE', aspect=1, height=5)
# sns.set(rc={'figure.facecolor':'white'})
# plt.show()


# Preprocessing data
# most likely catagories for separation after viewing full pairwise matrix data
choice = ['PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS', 'BALANCE', 'MINIMUM_PAYMENTS', 'CASH_ADVANCE']

# Select best data
df = pd.DataFrame(dfg[choice])

# fill nan with 0
df[['MINIMUM_PAYMENTS']] = df[['MINIMUM_PAYMENTS']].fillna(0)
df[['CREDIT_LIMIT']] = df[['CREDIT_LIMIT']].fillna(0)

df_scaled = pd.DataFrame(StandardScaler().fit_transform(df))
df_scaled.columns = choice

dfgY = dfg['CUST_ID']  # won't use this, but don't know how to leave it out at the moment
x_traindf, x_testdf, y_train, y_test = train_test_split(df_scaled, dfgY, test_size=0.1, random_state=42)
x_traindf.columns = choice
x_testdf.columns = choice
# x_test = torch.tensor(x_test)

#%% Elbow plot

clusters = []

kmax = 20

for i in range(1, kmax):
    km = KMeans(n_clusters=i).fit(x_train)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, kmax)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Annotate arrow
ax.annotate('Possible Elbow Point', xy=(4, 87800), xytext=(4, 97800), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible Elbow Point', xy=(10, 60000), xytext=(10, 70000), xycoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

fig.savefig('final//Kelbow.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

#%%
# do pca on training data
U, S, V = torch.pca_lowrank(torch.tensor(x_traindf.values), center=False)

# plot explained variance
EV = []
for i, s in enumerate(S):
    EV.append(s.numpy()/sum(S.numpy()))
    if i>=1:
        EV[i] += EV[i-1]

print(EV)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(list(range(1, len(EV)+1)), EV, linewidth=2, color='r')
ax.set_xlabel('$\lambda_i$', fontsize=16, fontweight='bold')
ax.set_ylabel('Percent of variance', fontsize=16, fontweight='bold')
ax.set_title('Explained Variance of Reduced Dataset', fontsize=24, fontweight='bold')
plt.show()


#%%
# do pca on training data
U, S, V = torch.pca_lowrank(torch.tensor(x_traindf.values), center=False)

rd = 5  # number of principal modes

# get projected data for model
x_train_proj = torch.tensor(x_traindf.values) @ V[:, :rd]

# Do Kmeans clustering algorithm
K = 6  # number of clusters
model = KMeans(n_clusters = K)
label = model.fit_predict(x_train_proj)

# create a 'cluster' column
# df_scaled['cluster'] = label
x_traindf.insert(6, 'Cluster', label, True)
choice.append('Cluster')

#%%
# make a Seaborn pairplot
g = sns.pairplot(x_traindf, hue='Cluster', palette='coolwarm', vars=choice)
g.fig.suptitle('Clustered PCA Data, K=5, r=5 : Training Set', fontsize=28, fontweight='bold', y=0.999)
g.fig.savefig('PCApredictTrain.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

#%%
# get projected data for test model
x_test_proj = torch.tensor(x_testdf.values) @ V[:, :rd]

# Do Kmeans clustering algorithm
testlabel = model.predict(x_test_proj)

x_testdf.insert(6, 'Cluster', testlabel, True)

# make a Seaborn pairplot
# fig, ax = plt.subplots()
g = sns.pairplot(x_testdf, hue='Cluster', palette='coolwarm', vars=choice)
g.fig.suptitle('Clustered PCA Data, K=5, r=5 : Testing Set', fontsize=28, fontweight='bold', y=0.999)
g.fig.savefig('PCApredictTest.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()
#%%
# do svd on training data
U, S, Vh = torch.linalg.svd(torch.tensor(x_traindf.values))
V = Vh.T

rd = 5  # number of singular vectors

# get projected data for model
x_train_proj = torch.tensor(x_traindf.values) @ V[:, :rd]

# Do Kmeans clustering algorithm
K = 6  # number of clusters
model = KMeans(n_clusters = K)
label = model.fit_predict(x_train_proj)

# create a 'cluster' column
x_traindf = x_traindf.drop(['Cluster'], axis=1)
x_traindf.insert(6, 'Cluster', label, True)
# choice.append('Cluster')

# #%%
# make a Seaborn pairplot
g = sns.pairplot(x_traindf, hue='Cluster', palette='viridis', vars=choice)
g.fig.suptitle('Clustered SVD Data, K=5, r=5 : Training Set', fontsize=28, fontweight='bold', y=0.999)
g.fig.savefig('SVDpredictTrain.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

#%%
# get projected data for test model
x_test_proj = torch.tensor(x_testdf.values) @ V[:, :rd]

# Do Kmeans clustering algorithm
testlabel = model.predict(x_test_proj)

x_testdf = x_testdf.drop(['Cluster'], axis=1)
x_testdf.insert(6, 'Cluster', testlabel, True)

# make a Seaborn pairplot
# fig, ax = plt.subplots()
g = sns.pairplot(x_testdf, hue='Cluster', palette='viridis', vars=choice)
g.fig.suptitle('Clustered SVD Data, K=5, r=5 : Testing Set', fontsize=28, fontweight='bold', y=0.999)
g.fig.savefig('SVDpredictTest.png', facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()

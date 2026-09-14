
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

# 0. Create dataset
X,Y = make_blobs(cluster_std=0.5,random_state=20,n_samples=1000,centers=5)
print(X.shape, Y.shape)

# Read data
# f = open("S.txt", "r")
# file = f.read()
# from io import StringIO   # StringIO behaves like a file object
# c = StringIO(file)
# data = np.loadtxt(c)
# Y = data
# print(X.shape, Y.shape)

f = open("X.txt", "r")
file = f.read()
from io import StringIO   # StringIO behaves like a file object
c = StringIO(file)
data = np.loadtxt(c)
X = data

GMM = GaussianMixture(n_components=5).fit(X) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_ 
covariances = GMM.covariances_

# Plot   
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])

# for m,c in zip(means,covariances):
#     multi_normal = multivariate_normal(mean=m,cov=c)
#     ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(X).reshape(len(X),len(X)),colors='black',alpha=0.3)
#     ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)

plt.show()



# # Stratch dataset to get ellipsoid data
# X = np.dot(X,np.random.RandomState(0).randn(2,2))

# x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
# XY = np.array([x.flatten(),y.flatten()]).T

# GMM = GaussianMixture(n_components=5).fit(X) # Instantiate and fit the model
# print('Converged:',GMM.converged_) # Check if the model has converged
# means = GMM.means_ 
# covariances = GMM.covariances_


# # Plot   
# fig = plt.figure(figsize=(10,10))
# ax0 = fig.add_subplot(111)
# ax0.scatter(X[:,0],X[:,1])
# # ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
# for m,c in zip(means,covariances):
#     multi_normal = multivariate_normal(mean=m,cov=c)
#     ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
#     ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    
# plt.show()
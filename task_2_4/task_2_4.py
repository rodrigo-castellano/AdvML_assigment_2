import numpy as np
from scipy.stats import multivariate_normal, poisson
import matplotlib.pyplot as plt


def EM(X, S, n_clusters, iter):

    # Initialize the mean, covariance, pi and the rates
    aux = []
    for i in range (n_clusters):
        aux.append(i+1)
    aux = np.array(aux)
    rates = (np.mean(S)*np.ones(n_clusters))/aux

    pi = np.ones(n_clusters)/n_clusters 
    mu = np.random.randint(min(X[:,0]),max(X[:,0]),size=(n_clusters,len(X[0]))) 
    cov = np.zeros((n_clusters,len(X[0]),len(X[0])))
    for D in range(n_clusters):
        np.fill_diagonal(cov[D],3)


    for i in range(iter):               

        # E_step from EM. Calculate responsabilities so as to update parameters in the M_step    

        r = np.zeros((len(X),len(cov)))       
        den = 0
        for k in range(n_clusters): 
            den += pi[k]*multivariate_normal(mean=mu[k],cov=cov[k]).pdf(X)*poisson(rates[k]).pmf(S[k])   
            
        for k in range(n_clusters):
            num = pi[k]*multivariate_normal(mean=mu[k],cov=cov[k]).pdf(X)*poisson(rates[k]).pmf(S[k])
            r[:,k] = num/den

                
        
        # M_step from EM. Calculate m_k. with m_k and the responsabilities, update the parameters pi, mu, sigma

        pi = []
        mu = []
        cov = []
        rate = []
        for k in range(len(r[0])):
            m_k = np.sum(r[:,k],axis=0)
            mu_k = np.sum(X*r[:,k].reshape(len(X),1),axis=0)*(1/m_k)
            mu.append(mu_k)
            cov.append(((1/m_k)*np.dot((np.array(r[:,k]).reshape(len(X),1)*(X-mu_k)).T,(X-mu_k))))
            pi.append(m_k/np.sum(r)) 
            rate.append((1/m_k)*np.sum(S*r[:,k].reshape(len(S),1),axis=0))

    return mu, cov, pi, rates


# Reference:  https://python-course.eu/expectation_maximization_and_gaussian_mixture_models.php#Unsupervised-learning:-Clustering:-Gaussian-Mixture-Models-(GMM)
# np.random.seed(100)

from io import StringIO   
f = open("X.txt", "r")
file = f.read()
c = StringIO(file)
data = np.loadtxt(c)
X = data

f = open("S.txt", "r")
file = f.read()  
c = StringIO(file)
data = np.loadtxt(c)
S = data

n_clusters = 3
iter = 50  
mu, cov, pi, rates = EM(X, S, n_clusters,iter)



# PLOT

# Create a grid for the plot 
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))  # It can also be done with linespace
XY = np.array([x.flatten(),y.flatten()]).T
colors = ['green', 'blue', 'red', 'orange','purple']
plt.scatter(X[:,0],X[:,1])
for i in range(n_clusters):
    normal = multivariate_normal(mean=mu[i],cov=cov[i])
    plt.contour(np.sort(X[:,0]),np.sort(X[:,1]),normal.pdf(XY).reshape(len(X),len(X)),colors=colors[i],linewidths=rates[i],alpha=0.3)


plt.show()

# Reference:  https://python-course.eu/expectation_maximization_and_gaussian_mixture_models.php#Unsupervised-learning:-Clustering:-Gaussian-Mixture-Models-(GMM)



import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt



def EM(X, S, n_clusters, iter):

    # Initialize the mean, covariance and pi
    pi = np.ones(n_clusters)/n_clusters 
    mu = np.random.randint(min(X[:,0]),max(X[:,0]),size=(n_clusters,len(X[0]))) 
    cov = np.zeros((n_clusters,len(X[0]),len(X[0])))
    for D in range(n_clusters):
        np.fill_diagonal(cov[D],3)
    log_likelihood = []
    # To avoid singularities
    threshold = 1e-8*np.identity(len(X[0]))

    for i in range(iter):               

        # E_step from EM. Calculate responsabilities so as to update parameters in the M_step    

        r = np.zeros((len(X),len(cov)))       
        den = 0
        for k in range(n_clusters): 
                den += pi[k]*multivariate_normal(mean=mu[k],cov=cov[k]+threshold).pdf(X)        

        for k in range(n_clusters):
            cov[k]+=threshold
            num = pi[k]*multivariate_normal(mean=mu[k],cov=cov[k]+threshold).pdf(X)
            r[:,k] = num/den
                
        
        # M_step from EM. Calculate m_k. with m_k and the responsabilities, update the parameters pi, mu, sigma

        pi = []
        mu = []
        cov = []
        for k in range(len(r[0])):
            m_k = np.sum(r[:,k],axis=0)

            mu_k = (1/m_k)*np.sum(X*r[:,k].reshape(len(X),1),axis=0)
            mu.append(mu_k)
            cov.append(((1/m_k)*np.dot((np.array(r[:,k]).reshape(len(X),1)*(X-mu_k)).T,(X-mu_k)))+threshold)
            pi.append(m_k/np.sum(r)) 
        
        
        # Calculate the likelihood
        log_likelihood.append(np.log(np.sum([k*multivariate_normal(mu[i],cov[j]).pdf(X) for k,i,j in zip(pi,range(len(mu)),range(len(cov)))])))

    return log_likelihood, mu, cov, pi, r



np.random.seed(100)

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
log_likelihood, mu, cov, pi, r = EM(X, S, n_clusters,iter)

# PLOT
# Create a grid for the plot 
x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T

plt.scatter(X[:,0],X[:,1])
for i in range(n_clusters):
    normal = multivariate_normal(mean=mu[i],cov=cov[i])
    plt.contour(np.sort(X[:,0]),np.sort(X[:,1]),normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)

plt.show()



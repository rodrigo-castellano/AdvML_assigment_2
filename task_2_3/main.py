import numpy as np
from  scipy.stats import gamma
from  scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def iterations (iter, a_0, b_0, lambda_0, mu_0, E_tau):
    # Update parameters
    a_N = a_0 + N/2
    mu_N = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N) 

    for i in range(iter):  
        lambda_N = (lambda_0 + N)*E_tau

        E_mu = mu_N
        E_mu2 = 1/lambda_N + E_mu**2
        expr_1 = np.sum(X*X) -2*np.sum(X)*np.sum(mu_N) + N*E_mu2
        expr_2 = lambda_0*( E_mu2 + mu_0**2-2*mu_0*E_mu)
        b_N = b_0 + 0.5*(expr_1 + expr_2)

        # Update tau
        E_tau = a_N/b_N
    return a_N, b_N, lambda_N, mu_N


# Algorithm to calculate the variational posterior by approximation, using mean field theory.
# Case of univariate gaussian with params mu and its prior mu=mu(mu_prime, lambda), tauand its prior tau=tau(a,b). 
# Assumption: q(mu,tau)=q(mu)q(tau). In reality it is q(mu|tau)q(tau).
# After analyzing the distr of the factors, we see that q(mu)->Normal(mu|mu_N,lambda_N^-1) and q(tau)->Gam(tau|a_N, b_N)

# Draw samples from a Gaussian
mu, tau = 0.2, 0.5 # mean and precision of the gaussian
N = 30
X = np.random.normal(mu, 1/tau**0.5, N)
X_mean = np.sum(X)/N

iter = 10
np.random.seed(0)

# Initialize the parameters of the priors
a_0 = 1
b_0 = 1
lambda_0 = 1
mu_0 = 0
E_tau = 1


a_N, b_N, lambda_N, mu_N = iterations (iter, a_0, b_0, lambda_0, mu_0, E_tau)

# Plot: On the x-axis mu, on the y-axis tau, on the z-axis q(mu,tau)
mu_range = np.linspace(0, 1.0, num=200)
tau_range = np.linspace(0, 1.0, num=200)



# R = len(mu_range)
# C = len(tau_range)
# Z = np.empty((R,C))

# cont_x = cont_y = 0
# for mu in mu_range: 
#     print(cont_x)
#     for tau in tau_range: 
#         Z[cont_y,cont_x] = norm.pdf(mu, mu_N, 1/lambda_N**0.5)*gamma.pdf(tau,a=a_N,loc=0,scale=1/b_N)  # norm(i, 1/lambda_N**0.5).pdf(mu)*gamma.pdf(j, a_N, scale=1/b_N)
#         cont_y += 1
#     cont_x += 1
#     cont_y = 0

# xticks = np.round(mu_range,3)
# yticks = np.round(tau_range,3)


# plot_ = plt.contour(xticks, yticks, Z)
# plt.show()



from itertools import product
X, Y = np.meshgrid(tau_range, mu_range)
N, M = len(X), len(Y)
Z = np.zeros((N, M))
for i,(x,y) in enumerate(product(tau_range,mu_range)):
    pos = np.hstack((x, y))
    tau = pos[0]
    mu = pos[1]
    Z[np.unravel_index(i, (N,M))] =  norm(mu_N, 1/lambda_N**0.5).pdf(mu)*gamma.pdf(tau, a_N, scale=1/b_N)
    
im = plt.imshow(Z,cmap='hsv', origin='lower', extent=(0, 1, 0, 1)) #extent = (left, right, bottom, top)
ax = plt.gca()
ax.grid(False)
plt.title("Factorized Approximation, until convergence")
plt.xlabel('mu')
plt.ylabel('tau')
plt.show()

xticks = np.round(mu_range,3)
yticks = np.round(tau_range,3)


# plot_ = plt.contour(xticks, yticks, Z)
# plt.show()


# Compare with the posterior

post_mu = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N)
post_lambda = lambda_0 + N
post_a = a_0 + N/2
post_b = b_0 + 0.5*np.sum(np.power(X - X_mean, 2)) + (lambda_0*N*(X_mean - mu_0)**2)/(2*(lambda_0 + N))

post_mu_range = np.linspace(0, 1.0, num=200)
post_tau_range = np.linspace(0, 1.0, num=200)

X_post, Y_post = np.meshgrid(post_tau_range, post_mu_range)
post_N, post_M = len(X_post), len(Y_post)
Z_post = np.zeros((post_N, post_M))
for i,(x,y) in enumerate(product(post_tau_range,post_mu_range)):
    pos = np.hstack((x, y))
    tau = pos[0]
    mu = pos[1]
    tau_sample = gamma.pdf(tau, post_a, scale=1/post_b)
    Z_post[np.unravel_index(i, (post_N,post_M))] =  norm(post_mu, 1/(post_lambda*tau)**0.5).pdf(mu)*tau_sample
    
im = plt.imshow(Z_post,cmap='hsv', origin='lower', extent=(0, 1, 0, 1))
ax = plt.gca()
ax.grid(False)
plt.title("Exact Posterior, case 3")
plt.xlabel('mu')
plt.ylabel('tau')
plt.show()














# plot_ = sns.heatmap(Z, xticklabels=xticks, yticklabels=yticks)

# for ind, label in enumerate(plot_.get_xticklabels()):
#     if ind % 5 == 0:  # every 10th label is kept
#         label.set_visible(True)
#     else:
#         label.set_visible(False)
# for ind, label in enumerate(plot_.get_yticklabels()):
#     if ind % 5 == 0:  # every 10th label is kept
#         label.set_visible(True)
#     else:
#         label.set_visible(False)
# plt.show()



# mean = [0.5,0.5]
# sigmaxy = sigmayx = 0
# sigmax = sigmay = 0.1
# rates = Z.ravel()
# X, Y = np.round(mu_range,3), np.round(tau_range,3)

# rv = stats.multivariate_normal([mean[0], mean[1]], [[sigmax, sigmaxy], [sigmaxy, sigmay]])        
# Z = rv.pdf(np.dstack((X, Y)))  
# print('Z,rates: ', Z.shape, rates.shape) 
# # plt.contour(X, Y, Z,  linewidths=rates[i], alpha=0.1)
# plt.contour(Z)
# plt.show()
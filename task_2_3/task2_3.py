import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  scipy.stats import gamma
from  scipy.stats import norm

def iterations(a_0, b_0, lambda_0, mu_0, X, X_mean):
    # Initial guess
    E_tau = 1
    mu_N = 0
    old_E_tau = 1-E_tau
    # Update parameters  
    while abs(old_E_tau-E_tau)>10**(-6):    

        a_N = a_0 + N/2 
        mu_N = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N) 
        lambda_N = (lambda_0 + N)*E_tau 

        E_mu = mu_N
        E_mu2 = 1/lambda_N + E_mu**2
        expr_1 = np.sum(X*X) -2*np.sum(X)*mu_N + N*E_mu2
        expr_2 = lambda_0*( E_mu2 + mu_0**2-2*mu_0*E_mu)
        b_N = b_0 + 0.5*(expr_1 + expr_2) 
        
        old_E_tau = E_tau
        # Update tau    
        E_tau = a_N/b_N

    return a_N, b_N, lambda_N, mu_N

def exact_posterior(mu, tau, a_0, b_0, lambda_0, mu_0, X, X_mean):
    a_N = a_0 + N/2 
    mu_N = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N) 
    lambda_N = (lambda_0 + N)

    expr_1 = np.sum(X*X) -2*np.sum(X)*X_mean + N*X_mean**2
    expr_2 = lambda_0*N*( X_mean**2 + mu_0**2-2*mu_0*X_mean)/(2*(lambda_0+N))
    b_N = b_0 + 0.5*(expr_1 + expr_2) 

    gamma_likalihood = gamma.pdf(tau, a_N, scale=1/b_N)
    gauss_likelihood = norm.pdf(mu,mu_N,1/(lambda_N*tau)**0.5)
    posterior = gamma_likalihood*gauss_likelihood

    return posterior


np.random.seed(10)
# Draw samples from a Gaussian
mu, tau = 0, 0.1 # mean and precision of the gaussian
N = 100
X = np.random.normal(mu, 1/tau**0.5, N)
X_mean = np.sum(X)/N

# Initialize the parameters of the priors
a_0 = 0
b_0 = 0
lambda_0 = 0
mu_0 = 0
a_N, b_N, lambda_N, mu_N = iterations(a_0, b_0, lambda_0, mu_0, X, X_mean)


# Plot: On the x-axis mu, on the y-axis tau, on the z-axis q(mu,tau)

# Approximate distribution

mu_range = np.linspace(mu-1/(tau**0.5), mu+1/(tau**0.5), num=100)
tau_range = np.linspace(0, 4, num=100)

mu_range = np.linspace(-5,5, num=100)
tau_range = np.linspace(-5, 5, num=100)

R = len(mu_range)
C = len(tau_range)
Z = np.empty((R,C))

cont_x = cont_y = 0
for mu in mu_range: 
    cont_y = 0
    print(cont_x)
    for tau in tau_range: 
        Z[cont_y,cont_x] = norm.pdf(mu, mu_N, 1/lambda_N**0.5)*gamma.pdf(tau,a=a_N,loc=0,scale=1/b_N)  
        cont_y += 1
    cont_x += 1
print('cont_x,cont_y',cont_x,cont_y)

xticks = np.round(mu_range,3)
yticks = np.round(tau_range,3)


plot_ = plt.contour(xticks, yticks, Z, colors='green')


# Exact posterior comparison

cont_x = cont_y = 0
for mu in mu_range: 
    cont_y = 0
    print(cont_x)
    for tau in tau_range: 
        Z[cont_y,cont_x] = exact_posterior( mu, tau, a_0, b_0, lambda_0, mu_0, X, X_mean)  
        cont_y += 1
    cont_x += 1
print('cont_x,cont_y',cont_x,cont_y)

xticks = np.round(mu_range,3)
yticks = np.round(tau_range,3)


plot_ = plt.contour(xticks, yticks, Z, colors='blue')
plt.xlabel('mu')
plt.ylabel('tau')
plt.ylim([0,.3])
plt.xlim([-.5,1])
plt.show()
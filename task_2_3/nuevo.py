# import numpy as np
# from scipy.stats import norm, gamma, multivariate_normal
# from itertools import product
# import matplotlib.pyplot as plt

# np.random.seed(1338)
# N = 30 # number of samples
# mu = 0.2
# tau = 0.5 #precision

# X_range = np.linspace(0, 1.0, num=N)
# X = np.random.normal(mu, 1/tau**0.5, N)
# X_mean = np.mean(X)



# #Assuming prior P(mu|tau) to be N(mu_0, (lambda_0*tau)⁻1)
# # and prior p(tau) to be Gamma(a_0, b_0)

# #help functions
# def get_mu_N(X_mean, N):
#     mu_N = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N)
#     return mu_N

# def get_lambda_N(E_tau, N):
#     lambda_N = (lambda_0 + N)*E_tau
#     return lambda_N

# def get_a_N(N):
#     a_N = a_0 + N/2
#     return a_N

# def get_b_N(X, E_mu, Var_mu):
#     b_N = b_0 + 0.5*get_E_mu_expression(X, E_mu, Var_mu)
#     return b_N

# def get_E_mu_expression(X, E_mu, Var_mu):
#     E_mu_2 = Var_mu + E_mu**2
#     term1 = np.power(X, 2) - 2*X*E_mu + E_mu_2
#     summed = np.sum(term1)
#     term2 = lambda_0*E_mu_2 - 2*E_mu*mu_0 + mu_0**2
#     E_mu_expression = summed + term2
#     return E_mu_expression



#     #Prior parameters
# lambda_0 = 1
# mu_0 = 0
# a_0 = 1
# b_0 = 1

# def re_estimate_param(iterations, E_tau_guess):

#     #initial guess
#     E_tau = E_tau_guess

#     for i in range(iterations):
#         mu_N = get_mu_N(X_mean, N)
#         lambda_N = get_lambda_N(E_tau, N)

#         a_N = get_a_N(N)
#         b_N = get_b_N(X, mu_N, 1/lambda_N)

#         #re-estimate
#         E_tau = a_N/b_N
        
#     return lambda_N, mu_N, a_N, b_N


# pixels = 50
# #q_mu is gaussian and q_tau is gamma(a,b)
# q_mu_prec, q_mu_mean, q_tau_a, q_tau_b = re_estimate_param(iterations = 10, E_tau_guess = 1)

# mu_range = np.linspace(0, 1.0, num=pixels)
# tau_range = np.linspace(0, 1.0, num=pixels)

# X, Y = np.meshgrid(tau_range, mu_range)
# N, M = len(X), len(Y)
# Z = np.zeros((N, M))
# for i,(x,y) in enumerate(product(tau_range,mu_range)):
#     pos = np.hstack((x, y))
#     tau = pos[0]
#     mu = pos[1]
#     Z[np.unravel_index(i, (N,M))] =  norm(q_mu_mean, 1/q_mu_prec**0.5).pdf(mu)*gamma.pdf(tau, q_tau_a, scale=1/q_tau_b)
    
# im = plt.imshow(Z,cmap='hsv', origin='lower', extent=(0, 1, 0, 1)) #extent = (left, right, bottom, top)
# ax = plt.gca()
# ax.grid(False)
# plt.title("Factorized Approximation, until convergence")
# plt.xlabel('mu')
# plt.ylabel('tau')
# plt.show()


# post_mu = (lambda_0*mu_0 + N*X_mean)/(lambda_0 + N)
# post_lambda = lambda_0 + N
# post_a = a_0 + N/2
# post_b = b_0 + 0.5*np.sum(np.power(X - X_mean, 2)) + (lambda_0*N*(X_mean - mu_0)**2)/(2*(lambda_0 + N))

# post_mu_range = np.linspace(0, 1.0, num=pixels)
# post_tau_range = np.linspace(0, 1.0, num=pixels)

# X_post, Y_post = np.meshgrid(post_tau_range, post_mu_range)
# post_N, post_M = len(X_post), len(Y_post)
# Z_post = np.zeros((post_N, post_M))
# for i,(x,y) in enumerate(product(post_tau_range,post_mu_range)):
#     pos = np.hstack((x, y))
#     tau = pos[0]
#     mu = pos[1]
#     tau_sample = gamma.pdf(tau, post_a, scale=1/post_b)
#     Z_post[np.unravel_index(i, (post_N,post_M))] =  norm(post_mu, 1/(post_lambda*tau)**0.5).pdf(mu)*tau_sample
    
# im = plt.imshow(Z_post,cmap='hsv', origin='lower', extent=(0, 1, 0, 1))
# ax = plt.gca()
# ax.grid(False)
# plt.title("Exact Posterior, case 3")
# plt.xlabel('mu')
# plt.ylabel('tau')
# plt.show()





import numpy as np
import seaborn as sbs
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Setting hyperparameters

def inference_model(data, tol, a_0, b_0, lambda_0, mu_0):
    
    data_mean = np.mean(data)
    data_sum_squared = np.dot(data,data)
    
    # Initializing parameters
    old_parameters = 0
    new_parameters = 1
    N = np.size(data)
    E_tao = 1
    mu_N = 0
    
    while np.linalg.norm(new_parameters - old_parameters) > tol:
        
        
        mu_N, lambda_N = VI_mu_update(data,
                                 data_mean, N, lambda_0, mu_0, E_tao)
        a_N, b_N = VI_tao_update(data, data_mean,
                                 N, lambda_0, mu_0, a_0, b_0, mu_N, lambda_N)
        E_tao = a_N/b_N
        old_parameters = new_parameters
        new_parameters = np.array([mu_N, E_tao])
        print('a,b,mu,lambda, mu_N:',a_N, b_N, mu_N, lambda_N, mu_N)
        
    return mu_N, lambda_N, a_N, b_N

    
def VI_mu_update(data, data_mean, N, lambda_0, mu_0, E_tao):
    """
    Parameter: data: Nx1 vector of observed variables
    Parameter: data_mean: mean of data vector
    Parameter: lamda_0: hyperparameter; Precision for prior on mu
    Parameter: mu_0: hyperparameter; Mean for prior on mu
    Parameter: E_tao: Expected value of tao
    """
    
    # Variational distriution for mu - Gaussian
    mu_N = (lambda_0*mu_0 + N*data_mean) / (lambda_0 + N)
    lambda_N = (lambda_0 + N)*E_tao
    
    return mu_N, lambda_N

def VI_tao_update(data, data_mean, N,
                 lambda_0, mu_0, a_0, b_0, mu_N, lambda_N):
    
    a_N = a_0 + N/2
    
    first_term = np.dot(data,data)\
         - 2*N*mu_N*data_mean + N*(1/lambda_N + mu_N**2)
    second_term = lambda_0*((1/lambda_N + mu_N**2) - 2*mu_0*mu_N + mu_0**2)
    b_N = b_0 + 0.5*(first_term + second_term)
    
    return a_N, b_N
    
def true_posterior():
    pass


def real_posterior(data, mu, tao, a_0, b_0, lambda_0, mu_0):
    N = np.size(data)
    data_mean = np.mean(data)
    
    a_post = a_0 + N/2
    b_post = b_0 + 0.5 * (np.dot(data,data) \
        + lambda_0*mu_0**2 - (N*data_mean + lambda_0*mu_0)**2/(N+lambda_0))
    
    mu_post = ( N * data_mean + lambda_0*mu_0 )/(N + lambda_0)
    tao_post = tao * (N + lambda_0)
    std_post = np.sqrt(1/tao_post)
    
    gamma_likalihood = stats.gamma.pdf(tao, a_post, loc=0, scale=1/b_post)
    gauss_likelihood = stats.norm.pdf(mu,mu_post,std_post)
    
    return gamma_likalihood * gauss_likelihood

def inferred_posterior(mu, tao, mu_N, lambda_N, a_N, b_N):
    std_N = np.sqrt(1/lambda_N)
    
    gamma_likelihood = stats.gamma.pdf(tao, a_N, loc=0, scale=1/b_N)
    gauss_likelihood = stats.norm.pdf(mu,mu_N,std_N)
    
    return gamma_likelihood * gauss_likelihood


# Constructing data
real_mu = 100
real_tao = 100
N = 1000
sigma = np.sqrt(1/real_tao)
data = np.random.normal(real_mu, sigma, N)

# Hyperparameters
a_0 = 0
b_0 = 0
lambda_0 = 0
mu_0 = 100

mu_N, lambda_N, a_N, b_N = inference_model(data,
     tol = 1e-30, a_0=a_0,b_0=b_0, lambda_0=lambda_0, mu_0=mu_0)

E_tao = a_N/b_N


plot_range_x = 0.3
plot_range_y = 20
mu_axis = np.arange(mu_N-plot_range_x, mu_N + plot_range_x, 0.03)
tao_axis = np.arange(E_tao-plot_range_y, E_tao + plot_range_y, 0.03)

nx = np.size(mu_axis)
ny = np.size(tao_axis)

[X,Y] = np.meshgrid(mu_axis, tao_axis)
Z_real = np.asarray([[real_posterior(data,
 mu, tao, a_0, b_0, lambda_0, mu_0) for tao in tao_axis] for mu in mu_axis])
Z_inferred = np.asarray([[inferred_posterior(mu,
 tao, mu_N, lambda_N, a_N, b_N) for tao in tao_axis] for mu in mu_axis])

sns.set_style('darkgrid')
fig, ax = plt.subplots(1, 1) 
# plots contour lines 

cntr1 = ax.contour(X, Y, Z_real.T, colors='k',
     levels=[0.1,0.5,0.8,1,2,3,4,7,10])
cntr2 = ax.contour(X, Y, Z_inferred.T, colors='C0',
     levels=[0.1,0.5,0.8,1,2,3,4,7,10])
h1,_ = cntr1.legend_elements()
h2,_ = cntr2.legend_elements()

ax.legend([h1[0], h2[0]],
     ['Real posterior', 'Inferred posterior'], fontsize = 13, loc=3)

plt.xlabel('Mu', fontsize=13)
plt.ylabel('Tao', fontsize=13)
plt.savefig('VI_plot_N_1000-mu_100-tau_100-mu0_100', dpi=300)
plt.show() 
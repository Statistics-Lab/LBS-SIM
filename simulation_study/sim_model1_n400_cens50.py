import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
import scipy as sp
from scipy.stats import norm
from scipy import special
from scipy import stats
from scipy.linalg import sqrtm
import seaborn as sns
import time

start_time = time.time()   # record the start time of the program

# ---------------------------------------- Define B-spline ----------------------------------------
# B-spline technique converts some nonparametric estimation problems into parametric ones
# define the B-spline class
class BS_curve:    # quasi-uniform B-spline
    def __init__(self, nodenum, k, w_beta, u0=0, u1=1):
        # nodenum+k-1 B-spline basis functions of degree k
        self.w_beta = w_beta
        self.nodenum = nodenum   # number of knots
        self.k = k   # degree
        self.u0 = u0   # domain: [u0, u1]
        self.u1 = u1
        if self.nodenum == 2:
            self.knots = np.hstack((np.repeat(self.u0 - 0.01, self.k + 1), np.repeat(self.u1 + 0.01, self.k + 1)))  # np.hstack stacks the arrays of the parameter tuple horizontally
        else:
            percentiles = np.arange(0, 101, 100 / (nodenum - 1))[1:-1]  # generate an arithmetic sequence, excluding the two endpoints
            self.knots = np.hstack((np.repeat(self.u0 - 0.01, self.k+1), np.percentile(np.sort(self.w_beta), percentiles), np.repeat(self.u1 + 0.01, self.k+1)))  # np.hstack stacks the arrays of the parameter tuple horizontally
        # take equally spaced percentile knots:
        # if self.nodenum == 2:
        #     percentiles = np.array([0, 100])
        # else:
        #     percentiles = np.arange(0, 101, 100 / (nodenum - 1))  # generate an arithmetic sequence
        # self.knots = np.hstack((np.repeat(self.u0, self.k), np.percentile(np.sort(self.w_beta),percentiles), np.repeat(self.u1, self.k)))  #np.hstack stacks the arrays of the parameter tuple horizontally

    def base_b_spline(self, i, k, u):  # define the basis function
        # the i-th B-spline basis function of degree k, i=0,1,...,m+k-2, k is a nonnegative integer, m+k-1 basis functions in total
        knots = self.knots
        if k == 0:
            if u >= knots[i] and u < knots[i + 1]:  # i ranges from 0 to m+0-2
                return 1
            else:
                return 0
        else:
            if knots[i + k] - knots[i] == 0:
                a1 = 0  # coefficient; set the whole term to zero when the denominator is zero
            else:
                a1 = (u - knots[i]) / (knots[i + k] - knots[i])
            if knots[i + k + 1] - knots[i + 1] == 0:
                a2 = 0
            else:
                a2 = (knots[i + k + 1] - u) / (knots[i + k + 1] - knots[i + 1])
            return a1 * self.base_b_spline(i,k-1,u) + a2 * self.base_b_spline(i+1,k-1,u)

    def base_b_spline_dao(self, i, k, u):  # define the first derivative of the basis function
        knots = self.knots
        if k % 1 != 0 or k < 1:
            print('k must be a positive integer greater than 1')
            return
        else:
            if knots[i + k] - knots[i] == 0:
                d_a1 = 0  # coefficient
            else:
                d_a1 = k / (knots[i + k] - knots[i])
            if knots[i + k + 1] - knots[i + 1] == 0:
                d_a2 = 0
            else:
                d_a2 = k / (knots[i + k + 1] - knots[i + 1])
            return d_a1 * (self.base_b_spline(i,k-1,u)) - d_a2 * (self.base_b_spline(i+1,k-1,u))

    def base_b_spline_dao2(self, i, k, u):  # define the second derivative of the basis function
        knots = self.knots
        if k % 1 != 0 or k < 2:
            print('k must be a positive integer greater than 2')
            return
        else:
            if (knots[i + k] - knots[i]) == 0:
                d2_a1 = 0  # coefficient
            else:
                d2_a1 = k / (knots[i + k] - knots[i])
            if knots[i + k + 1] - knots[i + 1] == 0:
                d2_a2 = 0
            else:
                d2_a2 = k / (knots[i + k + 1] - knots[i + 1])
            return d2_a1 * self.base_b_spline_dao(i,k-1,u) - d2_a2 * self.base_b_spline_dao(i+1,k-1,u)

# define the basis-function matrix for a 1-D array
def base_function(x):
    bs = BS_curve(nodenum, degree, x, min(x), max(x))
    base_function = []  # basis functions
    for x_i in x:
        base_function.append([bs.base_b_spline(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
    return np.array(base_function)   # n x (m+k-1)

# define the first-derivative matrix of the basis functions for a 1-D array
def base_function_dao(x):
    bs = BS_curve(nodenum, degree, x, min(x), max(x))
    base_function_dao = []  # basis functions
    for x_i in x:
        base_function_dao.append([bs.base_b_spline_dao(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
    return np.array(base_function_dao)  # n x (m+k-1)

# define the second-derivative matrix of the basis functions for a 1-D array
def base_function_dao2(x):
    bs = BS_curve(nodenum, degree, x, min(x), max(x))
    base_function_dao2 = []  # basis functions
    for x_i in x:
        base_function_dao2.append([bs.base_b_spline_dao2(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
    return np.array(base_function_dao2)  # n x (m+k-1)


# ---------------------------------------- True parameter values ---------------------------------------

gamma_true = np.array([1,0.6])
beta_true = np.array([1/np.sqrt(2),1/np.sqrt(2)])
alpha_true = np.array([0.7])

# ---------------------------------------- Generate predictors -----------------------------------------------
n = 400   # sample size
# generate the categorical predictor z
# np.random.seed(3)
z1 = np.random.binomial(1,0.5,n)
# np.random.seed(4)
z2 = np.random.binomial(1,0.5,n)
z = np.c_[z1,z2]   #n×z_num
z_num = z.shape[1]
print('Categorical predictor z = ',z,z.shape)

# generate the continuous variable w
np.random.seed(6)
w1 = np.random.normal(0,1,n)
np.random.seed(10)
w2 = np.random.normal(0,1,n)
w = np.c_[w1,w2]   #n×w_num
w_num = w.shape[1]
print('Continuous predictor w = ',w,w.shape)

# ---------------------------------------- Iterative estimation ---------------------------------------------------
m = 1000  # total number of outer-loop iterations
iter = 0   # number of iterations completed so far
nodenum = 4   # number of B-spline knots
degree = 3   # degree of the B-spline

# storage arrays: hold each estimated value; initial values are not stored
gamma_m = np.zeros((m,z_num))
beta_m = np.zeros((m,w_num))
alpha_m = np.zeros((m,1))
eta_m = np.zeros((m,nodenum + degree - 1))
censor_rate_m = np.zeros((m,1))
g_true_m = np.zeros((n,1))
g_mean_m = np.zeros((n,1))
g_pre_m = np.zeros((n,m))

# define the unknown smooth function g(x)
g_func = 'sin(x)'
def g_function(x):
    return sin(x)

while iter < m:   # the estimation loop runs m times; the error term is regenerated each time, producing a new y
    print('Iteration', iter+1, ':')
    # np.random.seed(9)
    N = np.random.normal(0,1,n)
    epsilon = 2*np.arcsinh(alpha_true * N * 0.5)   # n x 1
    print( '\nerror term epsilon range: [{},{}]'.format(min(epsilon),max(epsilon)))

    z_gamma = z @ gamma_true
    w_beta = w @ beta_true

    # generate the response y
    y = z @ gamma_true + g_function(w @ beta_true) + epsilon   # n x 1
    print('\nresponse variable y range: [{},{}]'.format(min(y),max(y)))   # keep y from becoming too large

    censor_value = 0.9
    ind = np.where(y > censor_value, 0, 1)   # indicator variable, n x 1
    y = np.where(y > censor_value, censor_value, y)
    censor_rate = (n - np.sum(ind, axis=0)) / n
    print("Censoring rate:", censor_rate)
    censor_rate_m[iter] = censor_rate

# ----------------------------------- Maximum likelihood estimation -- Newton iterations ------------------------------------
    # initial values are the preset true values
    gamma = gamma_true   # z_num x 1
    beta = beta_true   # w_num x 1
    alpha = alpha_true   # 1 x 1

    # initial value of eta by least squares
    eta_true = np.linalg.inv(base_function(w@beta_true).T @ base_function(w@beta_true)) @ (base_function(w@beta_true).T @ g_function(w@beta_true)).T   # (m+k-1) x 1
    eta = eta_true

    # initial values
    theta0 = np.hstack((gamma.T,beta.T,alpha.T,eta.T)).T
    print('Initial theta0 = ',theta0, theta0.shape)
    print('Initial gamma = ', gamma, gamma.shape, '\ninitial beta = ', beta, beta.shape, '\ninitial alpha =',alpha, alpha.shape, '\ninitial eta = ', eta, eta.shape)

    Newton_max_iter = 200
    for j in range(Newton_max_iter):
        # reparameterize beta via lambda
        lambd = beta[1:]/beta[0]   # (w_num-1) x 1
        lambd_1 = np.insert(lambd,0,1)   # a 1 is prepended to lambd
        betalambd = -np.outer(lambd,lambd_1)*(np.sum(lambd_1**2)**(-3/2)) + np.eye(w_num)[1:]*(np.sum(lambd_1**2)**(-1/2))   #(w_num-1)×w_num
        betalambd2 = -np.kron(lambd.reshape(w_num-1,1),np.eye(w_num)[1:])*(np.sum(lambd_1**2)**(-3/2)) -np.kron(np.eye(w_num-1).flatten().reshape((w_num-1)**2,1),lambd_1)*(np.sum(lambd_1**2)**(-3/2)) \
                     +np.kron(lambd.reshape(w_num-1,1),np.outer(lambd,lambd_1))*3*(np.sum(lambd_1**2)**(-5/2)) -np.kron(np.eye(w_num)[1:],lambd.reshape(w_num-1,1))*(np.sum(lambd_1**2)**(-3/2))   # (w_num-1)**2 x w_num

        theta = np.hstack((gamma.T, lambd.T, alpha.T, eta.T)).T
        print('theta at iteration start = ', theta, theta.shape)

        w_beta = w @ beta   # n x 1

        # common factors before taking derivatives
        mu = z @ gamma + base_function(w_beta) @ eta   # n x 1

        sinh1 = np.sinh(y - mu)
        cosh1 = np.cosh(y - mu)
        sinh2 = np.sinh((y - mu)/2)
        cosh2 = np.cosh((y - mu)/2)
        tanh2 = np.tanh((y - mu)/2)
        standardN = stats.norm(loc=0, scale=1)

        logd = np.log(standardN.pdf((2/alpha)*sinh2))-np.log(1-standardN.cdf((2/alpha)*sinh2))
        d = np.exp(logd)
        print('\nd range: [{},{}]'.format(min(d),max(d)))
        # if d contains NaN or inf, break the Newton iterations
        if np.isnan(d).any() or np.isinf(d).any():
            break

        logd_dao = np.log((-(2/alpha)*sinh2*standardN.pdf((2/alpha)*sinh2))*(1-standardN.cdf((2/alpha)*sinh2))+(standardN.pdf((2/alpha)*sinh2))**2)-np.log((1-standardN.cdf((2/alpha)*sinh2))**2)
        d_dao = np.exp(logd_dao)
        print('\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))
        # if d contains NaN or inf, break the Newton iterations
        if np.isnan(d_dao).any() or np.isinf(d_dao).any():
            break

        # first derivatives
        lgamma = z.T @ (ind*(-0.5*tanh2+(1/(alpha**2))*sinh1)) + z.T @ ((1-ind)*(d*(1/alpha)*cosh2))   # z_num x 1
        llambd = (betalambd @ w.T) @ (ind * ((-0.5*tanh2+(1/(alpha**2))*sinh1))*(base_function_dao(w_beta) @ eta))  + (betalambd @ w.T) @ ((1-ind) * (d*(1/alpha)*cosh2)*(base_function_dao(w_beta) @ eta))   # (w_num-1) x 1
        lalpha = (ind @ (((4/(alpha**2))*(sinh2**2))-1))*(1/alpha) + ((1-ind) @ (d*sinh2)) *(2/(alpha**2))   # 1 x 1
        leta = base_function(w_beta).T @ (ind * (-0.5*tanh2 + (1/(alpha**2))*sinh1 ))  + base_function(w_beta).T @ ((1-ind) * (d*(1/alpha)*cosh2))   # (m+k-1) x 1
        # second derivatives
        lgamma2 = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ z + z.T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ z   # z_num x z_num
        lgammalambd = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1)))*(base_function_dao(w_beta) @ eta)) @ w @ betalambd.T + z.T @ np.diag((1-ind)*(((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))*(base_function_dao(w_beta) @ eta)) @ w @ betalambd.T   # z_num x (w_num-1)
        lgammaalpha = z.T @ (ind * ((-2/(alpha**3))*sinh1)) + z.T @ ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d)))   # z_num x 1
        lgammaeta = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ base_function(w_beta) + z.T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ base_function(w_beta)   # z_num x (m+k-1)
        llambd2 = betalambd @ w.T @ np.diag(ind * ((((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))*((base_function_dao(w_beta) @ eta)**2))) @ w @ betalambd.T \
                  + betalambd @ w.T @ np.diag(ind * ((-0.5*tanh2 + (1/(alpha**2))*sinh1 )*(base_function_dao2(w_beta) @ eta))) @ w @ betalambd.T \
                  + ((betalambd2 @ w.T) @ (ind * (-0.5*tanh2 + (1/(alpha**2))*sinh1 )*(base_function_dao(w_beta) @ eta))).reshape(w_num-1,w_num-1) \
                  + betalambd @ w.T @ np.diag( (1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))*((base_function_dao(w_beta) @ eta)**2) ) @ w @ betalambd.T \
                  + betalambd @ w.T @ np.diag((1-ind) * ((1/alpha)*cosh2*d ) * (base_function_dao2(w_beta) @ eta)) @ w @ betalambd.T \
                  + ((betalambd2 @ w.T) @ ((1-ind) * ((1/alpha)*cosh2*d ) * (base_function_dao(w_beta) @ eta))).reshape(w_num-1,w_num-1)   # (w_num-1) x (w_num-1)
        llambdalpha = betalambd @ w.T @ (ind * ((-2/(alpha**3))*sinh1) *(base_function_dao(w_beta) @ eta)) + betalambd @ w.T @ ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d)) *(base_function_dao(w_beta) @ eta))   #(w_num-1) × 1
        llambdeta = betalambd @ w.T @ np.diag(ind * ((((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )) )*(base_function_dao(w_beta) @ eta))) @ base_function(w_beta) + betalambd @ w.T @ np.diag(ind*(-0.5*tanh2 + (1/(alpha**2))*sinh1)) @ base_function_dao(w_beta)  \
                    +betalambd @ w.T @ np.diag((1-ind) * ((((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) *(base_function_dao(w_beta) @ eta)) @ base_function(w_beta) + betalambd @ w.T @ np.diag((1-ind)*((1/alpha)*cosh2*d)) @ base_function_dao(w_beta)   # (w_num-1) x (m+k-1)
        lalpha2 = ind @ ((-12/(alpha**4))*(sinh2**2)+(1/(alpha**2))) + (1-ind) @ ((-d*(4/(alpha**3))*sinh2) - ((-4/(alpha**4))*(sinh2**2)*d_dao  ))   # 1 x 1
        lalphaeta = (ind * ((-2/(alpha**3))*sinh1)).T @ base_function(w_beta) + ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d))).T @ base_function(w_beta)   # 1 x (m+k-1)
        leta2 = base_function(w_beta).T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ base_function(w_beta) + base_function(w_beta).T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ base_function(w_beta)   # (m+k-1) x (m+k-1)


        ltheta = np.hstack((lgamma.T,llambd.T,lalpha.T,leta.T)).T
        print('ltheta =', ltheta, max(abs(ltheta)))
        ltheta2 = np.block([[lgamma2                    ,lgammalambd                ,lgammaalpha.reshape(z_num,1) ,lgammaeta],
                            [lgammalambd.T              ,llambd2                    ,llambdalpha.reshape(w_num-1,1) ,llambdeta],
                            [lgammaalpha.T.reshape(1,z_num) ,llambdalpha.T.reshape(1,w_num-1) ,lalpha2                  ,lalphaeta.reshape(1,nodenum+degree-1)],
                            [lgammaeta.T                ,llambdeta.T                ,lalphaeta.T.reshape(nodenum+degree-1,1) ,leta2]])   # (z_num+(w_num-1)+1+(m+k-1)) x (z_num+(w_num-1)+1+(m+k-1))

        theta1 = theta - np.linalg.inv(ltheta2) @ ltheta
        print('theta1= ',theta1, theta1.shape)

        gamma = theta1[0:z_num]
        lambd = theta1[z_num:z_num+w_num-1]
        beta = np.insert(lambd,0,1) / np.sqrt(1+np.sum(lambd**2))
        alpha = theta1[z_num+w_num-1:z_num+w_num]
        eta = theta1[z_num+w_num:]

        # SE = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
        # print('SE=', SE)
        likelihood_value = ind@(log(1/alpha)+log(np.cosh((y-(z@gamma+base_function(w@beta)@eta))/2))\
                           -(2/(alpha**2))*np.sinh((y-(z@gamma+base_function(w@beta)@eta))/2)**2)\
                           +(1-ind)@(log(1-standardN.cdf((2/alpha)*np.sinh((y-(z@gamma+base_function(w@beta)@eta))/2))))
        print('likelihood_value =', likelihood_value)
        # AIC = -2 * likelihood_value + 2 * len(theta1)
        # print('AIC=', AIC)

        thre = max(abs(theta1 - theta))
        print('Step size:', thre)
        if (thre > 1e+3) or (j == 50 and thre > 0.05) or (j == Newton_max_iter-1):
            break
        if thre < 1e-5:
            gamma_m[iter] = gamma
            beta_m[iter] = beta
            alpha_m[iter] = alpha
            eta_m[iter] = eta
            print('theta1 at convergence = ', np.hstack((gamma.T, beta.T, alpha.T, eta.T)).T )
            if max(abs(base_function(sort(w@beta))@eta - g_function(sort(w@beta_true)))) < 2\
                and max(base_function(sort(w@beta))@eta)-min(base_function(sort(w@beta))@eta) > 1.5:   # prevent extreme outliers in the figure
                u1 = np.sort(w @ beta_m[iter].T).flatten()   # n x 1
                g_pre_m[:,iter] = base_function(u1) @ eta_m[iter].T
                iter += 1
            break
            # u1 = np.sort(w @ beta_m[iter].T).flatten()  # n x 1
            # g_pre_m[:, iter] = base_function(u1) @ eta_m[iter].T
            # iter += 1
            # break
        else:
            print('theta1 when not converged = ', np.hstack((gamma.T, beta.T, alpha.T, eta.T)).T )


print('gamma_m = ', gamma_m, gamma_m.shape)
print('beta_m = ', beta_m, beta_m.shape)
print('alpha_m = ', alpha_m, alpha_m.shape)
print('eta_m = ', eta_m, eta_m.shape)


# -------------------------- Compute Mean, Bias, SD, and MSE -----------------------------------
gamma_95low = np.percentile(gamma_m, 2.5, axis=0)
gamma_95upper = np.percentile(gamma_m, 97.5, axis=0)
gamma_mean = np.mean(gamma_m, axis=0)
gamma_bias = gamma_mean - gamma_true
gamma_sd = np.sqrt(np.sum((gamma_m - gamma_mean)**2, axis=0)/m)
gamma_mse = np.sum((gamma_m - gamma_true)**2, axis=0)/m
print('gamma_95low =', gamma_95low, gamma_95low.shape)
print('gamma_95upper =', gamma_95upper, gamma_95upper.shape)
print('gamma_mean =', gamma_mean, gamma_mean.shape)
print('gamma_bias =', gamma_bias, gamma_bias.shape)
print('gamma_sd =', gamma_sd, gamma_sd.shape)
print('gamma_mse =', gamma_mse, gamma_mse.shape)


beta_95low = np.percentile(beta_m, 2.5, axis=0)
beta_95upper = np.percentile(beta_m, 97.5, axis=0)
beta_mean = np.mean(beta_m, axis=0)
beta_bias = beta_mean - beta_true
beta_sd = np.sqrt(np.sum((beta_m - beta_mean)**2, axis=0)/m)
beta_mse = np.sum((beta_m - beta_true)**2, axis=0)/m
print('beta_95low =', beta_95low, beta_95low.shape)
print('beta_95upper =', beta_95upper, beta_95upper.shape)
print('beta_mean =', beta_mean, beta_mean.shape)
print('beta_bias =', beta_bias, beta_bias.shape)
print('beta_sd =', beta_sd, beta_sd.shape)
print('beta_mse =', beta_mse, beta_mse.shape)


alpha_95low = np.percentile(alpha_m, 2.5, axis=0)
alpha_95upper = np.percentile(alpha_m, 97.5, axis=0)
alpha_mean = np.mean(alpha_m, axis=0)
alpha_bias = alpha_mean - alpha_true
alpha_sd = np.sqrt(np.sum((alpha_m - alpha_mean)**2, axis=0)/m)
alpha_mse = np.sum((alpha_m - alpha_true)**2, axis=0)/m
print('alpha_95low =', alpha_95low, alpha_95low.shape)
print('alpha_95upper =', alpha_95upper, alpha_95upper.shape)
print('alpha_mean =', alpha_mean, alpha_mean.shape)
print('alpha_bias =', alpha_bias, alpha_bias.shape)
print('alpha_sd =', alpha_sd, alpha_sd.shape)
print('alpha_mse =', alpha_mse, alpha_mse.shape)


eta_95low = np.percentile(eta_m, 2.5, axis=0)
eta_95upper = np.percentile(eta_m, 97.5, axis=0)
eta_mean = np.mean(eta_m, axis=0)
eta_bias = eta_mean - eta_true
eta_sd = np.sqrt(np.sum((eta_m - eta_mean)**2, axis=0)/m)
eta_mse = np.sum((eta_m - eta_true)**2, axis=0)/m
print('eta_95low =', eta_95low, eta_95low.shape)
print('eta_95upper =', eta_95upper, eta_95upper.shape)
print('eta_mean =', eta_mean, eta_mean.shape)
print('eta_bias =', eta_bias, eta_bias.shape)
print('eta_sd =', eta_sd, eta_sd.shape)
print('eta_mse =', eta_mse, eta_mse.shape)


mean_censor_rate = np.mean(censor_rate_m, axis=0)
mean_censor_rate = mean_censor_rate[0]
print('mean_censor_rate =', mean_censor_rate)

# -------------------------- Compute IABIAS and MISE for the single-index function -----------------------------------
u3 = np.sort(w @ beta_true.T)
g_true_m[:,0] = g_function(u3)
g_func_IABIAS = np.mean(np.mean(abs(g_pre_m-g_true_m),axis=0))
g_func_MISE = np.mean(np.mean((g_pre_m-g_true_m)**2,axis=0))

# ---------------------------------------- Plotting -----------------------------------------
plt.rcParams['axes.unicode_minus'] = False   # avoid the minus-sign rendering issue when saving figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.figure(dpi=600)

fig, ax = plt.subplots()
for fig_iter in range(m):
    u1 = np.sort(w @ beta_m[fig_iter].T).flatten()  # n x 1
    g_pre = base_function(u1) @ eta_m[fig_iter].T

    u2 = np.sort(w @ beta_mean.T)  # n x 1
    mean_g_pre = base_function(u2) @ eta_mean.T
    g_mean_m[:,0] = mean_g_pre

    u3 = np.sort(w @ beta_true.T)
    g_true_m[:,0] = g_function(u3)

    ax.set_title(f"model1, m={m}, n={n}, censor_rate={mean_censor_rate:.3f}")
    plt.xlabel('$\mathrm{W^T \hat{\\beta}}$')
    plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')
    # ax.plot(u1[:-1], g_pre[:-1], color=(0.7, 0.7, 0.7))
    # ax.plot(u2[:-1], mean_g_pre[:-1], '--b')  # plot of the estimated mean
    # ax.plot(u3[:-1], g_function(u3)[:-1], '-r')  # raw data plot
    ax.plot(u1, g_pre, color=(0.7, 0.7, 0.7))
    ax.plot(u2, mean_g_pre, '--b')  # plot of the estimated mean
    ax.plot(u3, g_function(u3), '-r')  # raw data plot

lines, = plt.plot([], [], '-r', label='Real curve')
lines2, = plt.plot([], [], '--b', label='Single index')
plt.legend(loc='upper left', frameon=False)
plt.show()


# ------------------------------------- Save data and figures -----------------------------------------
folder_name = os.path.join(os.path.dirname(__file__), 'results', f'model1,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}')
os.makedirs(folder_name, exist_ok=True)

# save figures
fig_path = os.path.join(folder_name, f'model1_sim_single_index,nodenum={nodenum},m={m},n={n},mean_censor_rate={mean_censor_rate:.3f}.png')
fig.savefig(fig_path, dpi=600)

# save variables
var_path = os.path.join(folder_name, f'model1_sim_variables,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}.xlsx')
var_data = np.hstack([z, w, g_true_m, g_mean_m])
var_df1 = pd.DataFrame(var_data, columns=['z1','z2','w1','w2','g_true','g_mean'])
var_df2 = pd.DataFrame(g_pre_m)
with pd.ExcelWriter(var_path, engine='xlsxwriter') as writer:
    var_df1.to_excel(writer, index=False, sheet_name='model1_variables')
    var_df2.to_excel(writer, index=False, sheet_name='g_pre')


# save data
data_path = os.path.join(folder_name, f'model1_sim_results,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}.xlsx')
df1 = pd.DataFrame({'sample_size_n':[n]+[None]*(m-1), 'num_reps_m':[m]+[None]*(m-1),'mean_single_index':[g_func]+[None]*(m-1),
        'nodenum':[nodenum]+[None]*(m-1), 'degree':[degree]+[None]*(m-1),
        'censor_value':[censor_value]+[None]*(m-1),'censor_rate':censor_rate_m[:,0],
        'mean_censor_rate':[mean_censor_rate]+[None]*(m-1),
        })

df2_data = np.hstack([gamma_m, beta_m, alpha_m, eta_m])
df2 = pd.DataFrame(df2_data, columns=['γ1','γ2','β1','β2','α',
                                      'η1','η2','η3','η4','η5','η6'])

max_len = max(len(gamma_true),len(beta_true),len(eta_true))
gamma_len = max_len-len(gamma_true)
beta_len = max_len-len(beta_true)
alpha_len = max_len-len(alpha_true)
eta_len = max_len-len(eta_true)
df3 = pd.DataFrame({'γ_95low':list(gamma_95low)+[None]*gamma_len, 'γ_95upper':list(gamma_95upper)+[None]*gamma_len, 'γ_true':list(gamma_true)+[None]*gamma_len, 'γ_mean':list(gamma_mean)+[None]*gamma_len, 'γ_bias':list(gamma_bias)+[None]*gamma_len, 'γ_sd':list(gamma_sd)+[None]*gamma_len,'γ_mse':list(gamma_mse)+[None]*gamma_len,
                    'β_95low':list(beta_95low)+[None]*beta_len,   'β_95upper':list(beta_95upper)+[None]*beta_len,   'β_true':list(beta_true)+[None]*beta_len,   'β_mean':list(beta_mean)+[None]*beta_len,   'β_bias':list(beta_bias)+[None]*beta_len,   'β_sd':list(beta_sd)+[None]*beta_len,  'β_mse':list(beta_mse)+[None]*beta_len,
                    'α_95low':list(alpha_95low)+[None]*alpha_len, 'α_95upper':list(alpha_95upper)+[None]*alpha_len, 'α_true':list(alpha_true)+[None]*alpha_len, 'α_mean':list(alpha_mean)+[None]*alpha_len, 'α_bias':list(alpha_bias)+[None]*alpha_len, 'α_sd':list(alpha_sd)+[None]*alpha_len,'α_mse':list(alpha_mse)+[None]*alpha_len,
                    'η_95low':list(eta_95low)+[None]*eta_len,     'η_95upper':list(eta_95upper)+[None]*eta_len,     'η_true':list(eta_true)+[None]*eta_len,     'η_mean':list(eta_mean)+[None]*eta_len,     'η_bias':list(eta_bias)+[None]*eta_len,     'η_sd':list(eta_sd)+[None]*eta_len,    'η_mse':list(eta_mse)+[None]*eta_len,
                    'g_func_IABIAS':[g_func_IABIAS]+[None]*(max_len-1), 'g_func_MISE':[g_func_MISE]+[None]*(max_len-1)
                    })

with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
    df1.to_excel(writer, index=False, sheet_name='model1_set')
    df2.to_excel(writer, index=False, sheet_name='estimates')
    df3.to_excel(writer, index=False, sheet_name='evaluation_metrics')


# ---------------------------------------- Compute the running time -----------------------------------------
end_time = time.time()   # record the end time of the program
run_time = end_time - start_time   # compute the running time
print("Running time:", int(run_time // 60), "min", run_time % 60, "s")
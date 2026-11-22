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
    if np.array_equal(x, w_beta):
        bs = BS_curve(nodenum1, degree1, x, min(x), max(x))
        base_function = []  # basis functions
        for x_i in x:
            base_function.append([bs.base_b_spline(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function)   # n x (m+k-1)
    elif np.array_equal(x, u_rho):
        bs = BS_curve(nodenum2, degree2, x, min(x), max(x))
        base_function = []  # basis functions
        for x_i in x:
            base_function.append([bs.base_b_spline(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function)  # n x (m+k-1)

# define the first-derivative matrix of the basis functions for a 1-D array
def base_function_dao(x):
    if np.array_equal(x, w_beta):
        bs = BS_curve(nodenum1, degree1, x, min(x), max(x))
        base_function_dao = []  # basis functions first derivative
        for x_i in x:
            base_function_dao.append([bs.base_b_spline_dao(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function_dao)  # n x (m+k-1)
    elif np.array_equal(x, u_rho):
        bs = BS_curve(nodenum2, degree2, x, min(x), max(x))
        base_function_dao = []  # basis functions first derivative
        for x_i in x:
            base_function_dao.append([bs.base_b_spline_dao(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function_dao)  # n x (m+k-1)

# define the second-derivative matrix of the basis functions for a 1-D array
def base_function_dao2(x):
    if np.array_equal(x, w_beta):
        bs = BS_curve(nodenum1, degree1, x, min(x), max(x))
        base_function_dao2 = []  # basis functions second derivative
        for x_i in x:
            base_function_dao2.append([bs.base_b_spline_dao2(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function_dao2)  # n x (m+k-1)
    elif np.array_equal(x, u_rho):
        bs = BS_curve(nodenum2, degree2, x, min(x), max(x))
        base_function_dao2 = []  # basis functions second derivative
        for x_i in x:
            base_function_dao2.append([bs.base_b_spline_dao2(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
        return np.array(base_function_dao2)  # n x (m+k-1)


# ---------------------------------------- True parameter values ---------------------------------------

gamma_true = np.array([1,0.6])   # z_num x 1
beta_true = np.array([1/np.sqrt(2),1/np.sqrt(2)])   # w_num x 1
s_true = np.array([0.3,-0.3])   # x_num x 1
rho_true = np.array([1/np.sqrt(5),2/np.sqrt(5)])   # u_num x 1

# ---------------------------------------- Generate predictors -----------------------------------------------
n = 200  # sample size
# generate the categorical predictor z
# np.random.seed(1)
z1 = np.random.binomial(1,0.5,n)
# np.random.seed(6)
z2 = np.random.binomial(1,0.5,n)
z = np.c_[z1,z2]
z_num = z.shape[1]
print('Categorical predictor z = ',z,z.shape)

# generate the continuous variable w
np.random.seed(5)
w1 = np.random.normal(0,1,n)
np.random.seed(9)
w2 = np.random.normal(0,1,n)
w = np.c_[w1,w2]
w_num = w.shape[1]
print('Continuous predictor w = ',w,w.shape)

# generate the categorical predictor x
x = z
x_num = x.shape[1]
print('Categorical predictor x = ',x,x.shape)

# generate the continuous variable u
u = w
u_num = u.shape[1]
print('Continuous predictor u = ',u,u.shape)


# ---------------------------------------- Iterative estimation -----------------------------------------------
m = 1000  # total number of outer-loop iterations
iter = 0  # number of iterations completed so far
nodenum1 = 4  # number of B-spline knots
degree1 = 3   # degree of the B-spline
nodenum2 = 4
degree2 = 3

# storage arrays: hold each estimated value; initial values are not stored
gamma_m = np.zeros((m,z_num))
beta_m = np.zeros((m,w_num))
s_m = np.zeros((m,x_num))
rho_m = np.zeros((m,u_num))
eta_m = np.zeros((m,nodenum1 + degree1 - 1))
zeta_m = np.zeros((m,nodenum2 + degree2 - 1))
censor_rate_m = np.zeros((m,1))
g_true_m = np.zeros((n,1))
g_mean_m = np.zeros((n,1))
g_pre_m = np.zeros((n,m))
alpha_true_m = np.zeros((n,1))
alpha_mean_m = np.zeros((n,1))
alpha_pre_m = np.zeros((n,m))


# define the unknown smooth function g(x)
g_func = 'sin(x)'
def g_function(x):
    return sin(x)
alpha_func = '-cos(x)-1.1'
def alpha_function(x):
    return -cos(x)-1.1

while iter < m:   # the estimation loop runs m times; the error term is regenerated each time, producing a new y
    print('Iteration', iter+1, ':')
    N = np.random.normal(0,1,n)
    # N = np.loadtxt(os.path.join(os.path.dirname(__file__), 'model2_N.txt'))
    epsilon = 2*np.arcsinh(np.exp(x@s_true + alpha_function(u@rho_true)) * N * 0.5)  # n x 1
    print('\nerror term epsilon range: [{},{}]'.format(min(epsilon),max(epsilon)))

    # print('shape-parameter part', np.exp(x@s_true + alpha_function(u@rho_true)))

    z_gamma = z @ gamma_true
    w_beta = w @ beta_true
    u_rho = u @ rho_true

    # generate the response y
    y = z @ gamma_true + g_function(w_beta) + epsilon   # n x 1
    print('\nresponse variable y range: [{},{}]'.format(min(y),max(y)))   # keep y from becoming too large

    censor_value = 2.1
    ind = np.where(y > censor_value, 0, 1)   # indicator variable, n×1
    y = np.where(y > censor_value, censor_value, y)
    censor_rate = (n - np.sum(ind, axis=0)) / n
    print("Censoring rate:", censor_rate)
    censor_rate_m[iter] = censor_rate

# ----------------------------------- Maximum likelihood estimation -- Newton iterations ------------------------------------
    # initial values are the preset true values
    gamma = gamma_true  # z_num x 1
    beta = beta_true  # w_num x 1
    lambd = beta[1:]/beta[0]
    s = s_true  # x_num x 1
    rho = rho_true  # u_num x 1
    tau = rho[1:]/rho[0]

    # initial values of eta and zeta by least squares
    eta_true = np.linalg.inv(base_function(w_beta).T @ base_function(w_beta)) @ (base_function(w_beta).T @ g_function(w_beta)).T  # (m1+k1-1) x 1
    eta = eta_true
    zeta_true = np.linalg.inv(base_function(u_rho).T @ base_function(u_rho)) @ (base_function(u_rho).T @ alpha_function(u_rho)).T  # (m2+k2-1) x 1
    zeta = zeta_true

    # initial values
    theta0 = np.hstack((gamma.T, beta.T, s.T, rho.T, eta.T, zeta.T)).T
    print('Initial theta0 = ',theta0, theta0.shape)
    print('Initial gamma = ', gamma, gamma.shape, '\ninitial beta = ', beta, beta.shape, '\ninitial s = ', s, s.shape, '\ninitial rho =',rho, rho.shape, '\ninitial eta = ', eta, eta.shape, '\nzinitial eta = ', zeta, zeta.shape)

    error = 0
    zhengti_max_iter = 100
    for zhengti in range(zhengti_max_iter):
        if error == 1:
            break

        Theta = np.hstack((gamma.T, lambd.T, s.T, tau.T, eta.T, zeta.T)).T
        # --------------------------------- Fix the mean part; Newton iterations for the shape parameters ---------------------------------
        Newton_max_iter = 200  # set the maximum Newton iterations to avoid an infinite loop
        for j in range(Newton_max_iter):
            print('Shape parameter, iteration:', j)
            # reparameterize rho
            tau = rho[1:]/rho[0]
            tau_1 = np.insert(tau,0,1)  #a 1 is prepended to tau
            rhotau = -np.outer(tau,tau_1)*(np.sum(tau_1**2)**(-3/2)) + np.eye(u_num)[1:]*(np.sum(tau_1**2)**(-1/2))
            rhotau2 = -np.kron(tau.reshape(u_num-1,1),np.eye(u_num)[1:])*(np.sum(tau_1**2)**(-3/2)) -np.kron(np.eye(u_num-1).flatten().reshape((u_num-1)**2,1),tau_1)*(np.sum(tau_1**2)**(-3/2)) \
                         +np.kron(tau.reshape(u_num-1,1),np.outer(tau,tau_1))*3*(np.sum(tau_1**2)**(-5/2)) -np.kron(np.eye(u_num)[1:],tau.reshape(u_num-1,1))*(np.sum(tau_1**2)**(-3/2))

            theta = np.hstack((s.T, tau.T, zeta.T)).T
            print('Shape-parameter theta =', theta, theta.shape)

            w_beta = w @ beta  # n x 1
            u_rho = u @ rho  # n x 1

            # common factors before taking derivatives
            v = np.exp( x @ s + base_function(u_rho) @ zeta)   # n x 1
            print('\nv range: [{},{}]'.format(min(v),max(v)))
            # if any(list(num <= 0 or num > 2 for num in v)):   # the generator expression is a Python generator object and needs conversion via list
            #     error = 1
            #     break

            mu = z @ gamma + base_function(w_beta) @ eta   # n x 1
            sinh1 = np.sinh(y - mu)
            cosh1 = np.cosh(y - mu)
            sinh2 = np.sinh((y - mu)/2)
            cosh2 = np.cosh((y - mu)/2)
            tanh2 = np.tanh((y - mu)/2)
            standardN = stats.norm(loc=0, scale=1)

            logd = np.log(standardN.pdf((2/v)*sinh2))-np.log(1-standardN.cdf((2/v)*sinh2))
            d = np.exp(logd)
            print('\nd range: [{},{}]'.format(min(d),max(d)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d).any() or np.isinf(d).any():
                error = 1
                break

            logd_dao = np.log((-(2/v)*sinh2*standardN.pdf((2/v)*sinh2))*(1-standardN.cdf((2/v)*sinh2))+(standardN.pdf((2/v)*sinh2))**2)-np.log((1-standardN.cdf((2/v)*sinh2))**2)
            d_dao = np.exp(logd_dao)
            print('\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d_dao).any() or np.isinf(d_dao).any():
                error = 1
                break

            # first derivatives
            ls = x.T @ (ind * ((4/(v**2))*(sinh2**2)-1)) + x.T @  ((1-ind) * (d*sinh2*(2/v)))   # x_num x 1
            ltau = (rhotau @ u.T) @ (ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao(u_rho)@zeta))\
                 + (rhotau @ u.T) @ ((1-ind) * (d*sinh2*(2/v)) * (base_function_dao(u_rho)@zeta))   # (u_num-1) x 1
            lzeta = base_function(u_rho).T @ (ind * ((4/(v**2))*(sinh2**2)-1))\
                  + base_function(u_rho).T @ ((1-ind) * (d*sinh2*(2/v)))   # (m2+k2-1) x 1

            # second derivatives
            ls2 = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @x \
                + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @x   # x_num x x_num
            lstau = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * (base_function_dao(u_rho)@zeta)) @ u @ rhotau.T \
                  + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * (base_function_dao(u_rho)@zeta)) @ u @ rhotau.T   # x_num x (u_num-1)
            lszeta = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @base_function(u_rho) \
                + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @base_function(u_rho)   # x_num x (m2+k2-1)


            ltau2 = rhotau @ u.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * ((base_function_dao(u_rho)@zeta)**2)) @u@rhotau.T \
                  + rhotau @ u.T @ np.diag(ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao2(u_rho)@zeta)) @u@rhotau.T \
                  + ((rhotau2@u.T) @ (ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao(u_rho)@zeta))).reshape(u_num-1,u_num-1) \
                  + rhotau @ u.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * ((base_function_dao(u_rho)@zeta)**2)) @u@rhotau.T \
                  + rhotau @ u.T @ np.diag((1-ind) * (d*sinh2*(2/v)) * (base_function_dao2(u_rho)@zeta)) @u@rhotau.T \
                  + ((rhotau2@u.T) @ ((1-ind)*(d*sinh2*(2/v)) * (base_function_dao(u_rho)@zeta))).reshape(u_num-1,u_num-1)   # (u_num-1) x (u_num-1)
            ltauzeta = rhotau @ u.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * (base_function_dao(u_rho)@zeta)) @ base_function(u_rho) \
                     + rhotau @ u.T @ np.diag(ind * ((4/(v**2))*(sinh2**2)-1)) @ base_function_dao(u_rho) \
                     + rhotau @ u.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * (base_function_dao(u_rho)@zeta)) @ base_function(u_rho) \
                     + rhotau @ u.T @ np.diag((1-ind) * (d*sinh2*(2/v))) @ base_function_dao(u_rho)   # (u_num-1) x (m2+k2-1)


            lzeta2 = base_function(u_rho).T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @base_function(u_rho) \
                   + base_function(u_rho).T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @base_function(u_rho)   # (m2+k2-1) x (m2+k2-1)


            ltheta = np.hstack((ls.T,ltau.T,lzeta.T)).T  # (2+1+2+1+m1+k1-1+m2+k2-1) x 1
            print('ltheta =', ltheta, ltheta.shape)
            if np.isnan(ltheta).any() or np.isinf(ltheta).any():
                error = 1
                break

            ltheta2 = np.block([[ls2,      lstau,       lszeta],
                               [lstau.T,   ltau2,       ltauzeta],
                               [lszeta.T,  ltauzeta.T,  lzeta2]])
            theta1 = theta - (np.linalg.inv(ltheta2) @ ltheta)   # (2+1+2+1+m1+k1-1+m2+k2-1) x 1
            print('Shape-parameter theta1 = ', theta1, theta1.shape)

            s = theta1[0:x_num]
            tau = theta1[x_num:x_num+u_num-1]
            rho = np.insert(tau,0,1) / np.sqrt(1+np.sum(tau**2))
            zeta = theta1[x_num+u_num-1:]

            thre = max(abs(theta1 - theta))
            print('Shape-parameter step size:', thre)
            if (thre > 1e+3) or (j > 60 and thre > 0.05) or (j == Newton_max_iter - 1):
                error = 2
                break
            elif thre < 1e-5:
                print('Shape-parameter theta1 at convergence = ', np.hstack((s.T, rho.T, zeta.T)).T)
                break
            else:
                print('Shape-parameter theta1 when not converged = ', np.hstack((s.T, rho.T, zeta.T)).T )
        if error == 1:
            break
        # ---------------------------------- Fix the shape-parameter part; Newton iterations for the mean parameters ------------------------------------
        Newton_max_iter = 200
        for j in range(Newton_max_iter):
            print('Mean parameter, iteration:', j)
            # reparameterize beta
            lambd = beta[1:]/beta[0]
            lambd_1 = np.insert(lambd,0,1)  # a 1 is prepended to lambd
            betalambd = -np.outer(lambd,lambd_1)*(np.sum(lambd_1**2)**(-3/2)) + np.eye(w_num)[1:]*(np.sum(lambd_1**2)**(-1/2))
            betalambd2 = -np.kron(lambd.reshape(w_num-1,1),np.eye(w_num)[1:])*(np.sum(lambd_1**2)**(-3/2)) -np.kron(np.eye(w_num-1).flatten().reshape((w_num-1)**2,1),lambd_1)*(np.sum(lambd_1**2)**(-3/2)) \
                         +np.kron(lambd.reshape(w_num-1,1),np.outer(lambd,lambd_1))*3*(np.sum(lambd_1**2)**(-5/2)) -np.kron(np.eye(w_num)[1:],lambd.reshape(w_num-1,1))*(np.sum(lambd_1**2)**(-3/2))

            theta = np.hstack((gamma.T, lambd.T, eta.T)).T
            print('Mean-parameter theta =', theta, theta.shape)

            w_beta = w @ beta  # n x 1
            u_rho = u @ rho  # n x 1

            # common factors before taking derivatives
            v = np.exp( x @ s + base_function(u_rho) @ zeta)   # n x 1
            print('\nv range: [{},{}]'.format(min(v),max(v)))
            # if any(list(num <= 0 or num > 2 for num in v)):   # the generator expression is a Python generator object and needs conversion via list
            #     error = 1
            #     break

            mu = z @ gamma + base_function(w_beta) @ eta   # n x 1
            sinh1 = np.sinh(y - mu)
            cosh1 = np.cosh(y - mu)
            sinh2 = np.sinh((y - mu)/2)
            cosh2 = np.cosh((y - mu)/2)
            tanh2 = np.tanh((y - mu)/2)
            standardN = stats.norm(loc=0, scale=1)

            logd = np.log(standardN.pdf((2/v)*sinh2))-np.log(1-standardN.cdf((2/v)*sinh2))
            d = np.exp(logd)
            print('\nd range: [{},{}]'.format(min(d),max(d)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d).any() or np.isinf(d).any():
                error = 1
                break

            logd_dao = np.log((-(2/v)*sinh2*standardN.pdf((2/v)*sinh2))*(1-standardN.cdf((2/v)*sinh2))+(standardN.pdf((2/v)*sinh2))**2)-np.log((1-standardN.cdf((2/v)*sinh2))**2)
            d_dao = np.exp(logd_dao)
            print('\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d_dao).any() or np.isinf(d_dao).any():
                error = 1
                break

            # first derivatives
            lgamma = z.T @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1)) + z.T @ ((1-ind) * (d*(1/v)*cosh2))   # z_num x 1
            llambd = (betalambd @ w.T) @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao(w_beta)@eta)) \
                   + (betalambd @ w.T) @ ((1-ind) * (d*(1/v)*cosh2) * (base_function_dao(w_beta)@eta))   # (w_num-1) x 1
            leta = base_function(w_beta).T @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1))\
                 + base_function(w_beta).T @ ((1-ind) * (d*(1/v)*cosh2))   # (m1+k1-1) x 1

            # second derivatives
            lgamma2 = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ z \
                    + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ z   # z_num x z_num
            lgammalambd = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * (base_function_dao(w_beta)@eta)) @ w @ betalambd.T \
                        + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * (base_function_dao(w_beta) @ eta)) @ w @ betalambd.T   # z_num x (w_num-1)
            lgammaeta = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ base_function(w_beta) \
                      + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ base_function(w_beta)   #z_num × (m1+k1-1)


            llambd2 = betalambd @ w.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * ((base_function_dao(w_beta)@eta)**2)) @ w @ betalambd.T \
                    + betalambd @ w.T @ np.diag(ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao2(w_beta)@eta)) @ w @ betalambd.T \
                    + ((betalambd2@w.T) @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao(w_beta)@eta))).reshape(w_num-1,w_num-1) \
                    + betalambd @ w.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * ((base_function_dao(w_beta)@eta)**2)) @ w @ betalambd.T \
                    + betalambd @ w.T @ np.diag((1-ind) * ((1/v)*cosh2*d) * (base_function_dao2(w_beta)@eta)) @ w @ betalambd.T \
                    + ((betalambd2@w.T) @ ((1-ind) * ((1/v)*cosh2*d) * (base_function_dao(w_beta)@eta))).reshape(w_num-1,w_num-1)   #(w_num-1)×(w_num-1)
            llambdeta = betalambd @ w.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * (base_function_dao(w_beta) @ eta)) @ base_function(w_beta) \
                      + betalambd @ w.T @ np.diag(ind * (-0.5*tanh2+(1/(v**2))*sinh1)) @ base_function_dao(w_beta)  \
                      + betalambd @ w.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * (base_function_dao(w_beta) @ eta)) @ base_function(w_beta) \
                      + betalambd @ w.T @ np.diag((1-ind) * ((1/v)*cosh2*d)) @ base_function_dao(w_beta)    #(w_num-1) × (m1+k1-1)


            leta2 = base_function(w_beta).T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ base_function(w_beta) \
                  + base_function(w_beta).T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ base_function(w_beta)   # (m1+k1-1) x (m1+k1-1)


            ltheta = np.hstack((lgamma.T,llambd.T,leta.T)).T  # (2+1+2+1+m1+k1-1+m2+k2-1) x 1
            print('ltheta =', ltheta, ltheta.shape)
            if np.isnan(ltheta).any() or np.isinf(ltheta).any():
                error = 1
                break

            ltheta2 = np.block([[lgamma2,          lgammalambd,       lgammaeta],
                                [lgammalambd.T,    llambd2,           llambdeta],
                                [lgammaeta.T,      llambdeta.T,      leta2]])

            theta1 = theta - (np.linalg.inv(ltheta2) @ ltheta)   # (2+1+2+1+m1+k1-1+m2+k2-1) x 1
            print('theta1= ',theta1, theta1.shape)

            gamma = theta1[0:z_num]
            lambd = theta1[z_num:z_num+w_num-1]
            beta = np.insert(lambd,0,1) / np.sqrt(1+np.sum(lambd**2))
            eta = theta1[z_num+w_num-1:]

            thre = max(abs(theta1 - theta))
            print('Mean-parameter step size:', thre)
            if (thre > 1e+3) or (j > 60 and thre > 0.05) or (j == Newton_max_iter - 1):
                error = 1
                break
            elif thre < 1e-5:
                print('theta1 at convergence = ', np.hstack((gamma.T, beta.T, eta.T)).T)
                break
            else:
                print('theta1 when not converged = ', np.hstack((gamma.T, beta.T, eta.T)).T )
        if error == 1:
            break


        Theta1 = np.hstack((gamma.T, lambd.T, s.T, tau.T, eta.T, zeta.T)).T
        print('Overall Theta1 =', Theta1)
        likelihood_value = (ind@(log(1/sqrt(2*pi))+log(1/v)+log(np.cosh((y-(z@gamma+base_function(w_beta)@eta))/2))\
                           -(2/(v**2))*np.sinh((y-(z@gamma+base_function(w_beta)@eta))/2)**2)\
                           +(1-ind)@(log(1-standardN.cdf((2/v)*np.sinh((y-(z@gamma+base_function(w_beta)@eta))/2)))) )
        print('likelihood_value =', likelihood_value)
        # AIC = -2*likelihood_value+2*len(Theta1)
        # print('AIC =', AIC)

        Thre = max(abs(Theta1 - Theta))
        print('Overall step size:', Thre)
        if (Thre > 1e+3) or (zhengti > 60 and Thre > 0.05) or (zhengti == zhengti_max_iter-1):
            break
        if Thre < 1e-5:
            gamma_m[iter] = gamma  # store the estimate into the pre-allocated storage array
            beta_m[iter] = beta
            s_m[iter] = s
            rho_m[iter] = rho
            eta_m[iter] = eta
            zeta_m[iter] = zeta

            print('Theta1 at convergence =', Theta1)
            print('Final gamma =', gamma)
            print('Final beta =', beta)
            print('Final eta =', eta)
            print('Final s =', s)
            print('Final rho =', rho)
            print('Final zeta =', zeta)

            w_beta = np.sort(w@beta)
            u_rho = np.sort(u@rho)
            if max(abs(base_function(w_beta)@eta - g_function(sort(w@beta_true)))) < 2 \
                and max(abs(base_function(u_rho)@zeta - alpha_function(sort(u@rho_true)))) < 2 \
                and max(base_function(w_beta)@eta) - min(base_function(w_beta)@eta) > 1.5 \
                and max(base_function(u_rho)@zeta) - min(base_function(u_rho)@zeta) > 1.5:   # remove outliers
                w_beta = np.sort(w @ beta_m[iter].T).flatten()
                g_pre_m[:,iter] = base_function(w_beta) @ eta_m[iter].T

                u_rho = np.sort(u @ rho_m[iter].T).flatten()
                alpha_pre_m[:,iter] = base_function(u_rho) @ zeta_m[iter].T
                iter += 1
            break

            # w_beta = np.sort(w @ beta_m[iter].T).flatten()
            # g_pre_m[:, iter] = base_function(w_beta) @ eta_m[iter].T
            # u_rho = np.sort(u @ rho_m[iter].T).flatten()
            # alpha_pre_m[:, iter] = base_function(u_rho) @ zeta_m[iter].T
            # iter += 1
            # break



print('gamma_m = ', gamma_m, gamma_m.shape)
print('beta_m = ', beta_m, beta_m.shape)
print('s_m = ', s_m, s_m.shape)
print('rho_m = ', rho_m, rho_m.shape)
print('eta_m = ', eta_m, eta_m.shape)
print('zeta_m = ', zeta_m, zeta_m.shape)


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


s_95low = np.percentile(s_m, 2.5, axis=0)
s_95upper = np.percentile(s_m, 97.5, axis=0)
s_mean = np.mean(s_m, axis=0)
s_bias = s_mean - s_true
s_sd = np.sqrt(np.sum((s_m - s_mean)**2, axis=0)/m)
s_mse = np.sum((s_m - s_true)**2, axis=0)/m
print('s_95low =', s_95low, s_95low.shape)
print('s_95upper =', s_95upper, s_95upper.shape)
print('s_mean =', s_mean, s_mean.shape)
print('s_bias =', s_bias, s_bias.shape)
print('s_sd =', s_sd, s_sd.shape)
print('s_mse =', s_mse, s_mse.shape)


rho_95low = np.percentile(rho_m, 2.5, axis=0)
rho_95upper = np.percentile(rho_m, 97.5, axis=0)
rho_mean = np.mean(rho_m, axis=0)
rho_bias = rho_mean - rho_true
rho_sd = np.sqrt(np.sum((rho_m - rho_mean)**2, axis=0)/m)
rho_mse = np.sum((rho_m - rho_true)**2, axis=0)/m
print('rho_95low =', rho_95low, rho_95low.shape)
print('rho_95upper =', rho_95upper, rho_95upper.shape)
print('rho_mean =', rho_mean, rho_mean.shape)
print('rho_bias =', rho_bias, rho_bias.shape)
print('rho_sd =', rho_sd, rho_sd.shape)
print('rho_mse =', rho_mse, rho_mse.shape)


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


zeta_95low = np.percentile(zeta_m, 2.5, axis=0)
zeta_95upper = np.percentile(zeta_m, 97.5, axis=0)
zeta_mean = np.mean(zeta_m, axis=0)
zeta_bias = zeta_mean - zeta_true
zeta_sd = np.sqrt(np.sum((zeta_m - zeta_mean)**2, axis=0)/m)
zeta_mse = np.sum((zeta_m - zeta_true)**2, axis=0)/m
print('zeta_95low =', zeta_95low, zeta_95low.shape)
print('zeta_95upper =', zeta_95upper, zeta_95upper.shape)
print('zeta_mean =', zeta_mean, zeta_mean.shape)
print('zeta_bias =', zeta_bias, zeta_bias.shape)
print('zeta_sd =', zeta_sd, zeta_sd.shape)
print('zeta_mse =', zeta_mse, zeta_mse.shape)


mean_censor_rate = np.mean(censor_rate_m, axis=0)
mean_censor_rate = mean_censor_rate[0]
print('mean_censor_rate =', mean_censor_rate)


# -------------------------- Compute IABIAS and MISE for the single-index function -----------------------------------
u3 = np.sort(w @ beta_true.T)
g_true_m[:,0] = g_function(u3)
g_func_IABIAS = np.mean(np.mean(abs(g_pre_m-g_true_m),axis=0))
g_func_MISE = np.mean(np.mean((g_pre_m-g_true_m)**2,axis=0))

u6 = np.sort(u @ rho_true.T)
alpha_true_m[:,0] = alpha_function(u6)
alpha_func_IABIAS = np.mean(np.mean(abs(alpha_pre_m-alpha_true_m),axis=0))
alpha_func_MISE = np.mean(np.mean((alpha_pre_m-alpha_true_m)**2,axis=0))

# ---------------------------------------- Plotting -----------------------------------------
plt.rcParams['axes.unicode_minus'] = False   # avoid the minus-sign rendering issue when saving figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.figure(dpi=600)

fig1, ax = plt.subplots()
for fig_g_iter in range(m):
    w_beta = np.sort(w @ beta_m[fig_g_iter].T).flatten()  
    u1 = np.sort(w @ beta_m[fig_g_iter].T).flatten()  
    g_pre = base_function(w_beta) @ eta_m[fig_g_iter].T

    w_beta = np.sort(w @ beta_mean.T)  
    u2 = np.sort(w @ beta_mean.T)  
    mean_g_pre = base_function(w_beta) @ eta_mean.T
    g_mean_m[:,0] = mean_g_pre

    u3 = np.sort(w @ beta_true.T)
    g_true_m[:,0] = g_function(u3)

    ax.set_title(f"model2_mean, m={m}, n={n}, censor_rate={mean_censor_rate:.3f}")
    plt.xlabel('$\mathrm{W^T\hat{\\beta}}$')
    plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')
    # ax.plot(u1[:-1], g_pre[:-1], color=(0.7, 0.7, 0.7))
    # ax.plot(u2[:-1], mean_g_pre[:-1], '--b')   
    # ax.plot(u3[:-1], g_function(u3)[:-1], '-r')
    ax.plot(u1, g_pre, color=(0.7, 0.7, 0.7))
    ax.plot(u2, mean_g_pre, '--b')
    ax.plot(u3, g_function(u3), '-r')

lines, = plt.plot([], [], '-r', label='Real curve')
lines2, = plt.plot([], [], '--b', label='Single index')
plt.legend(loc='upper left', frameon=False)
plt.show()


fig2, ax = plt.subplots()
for fig_alpha_iter in range(m):
    u_rho = np.sort(u @ rho_m[fig_alpha_iter].T).flatten()
    u4 = np.sort(u @ rho_m[fig_alpha_iter].T).flatten()  # n x 1
    alpha_pre = base_function(u_rho) @ zeta_m[fig_alpha_iter].T

    u_rho = np.sort(u @ rho_mean.T)
    u5 = np.sort(u @ rho_mean.T)  # n x 1
    mean_alpha_pre = base_function(u_rho) @ zeta_mean.T
    alpha_mean_m[:,0] = mean_alpha_pre

    u6 = np.sort(u @ rho_true.T)  # n x 1
    alpha_true_m[:,0] = alpha_function(u6)

    ax.set_title(f"model2_shape, m={m}, n={n}, censor_rate={mean_censor_rate:.3f}")
    plt.xlabel('$\mathrm{U^T\hat{\\rho}}$')
    plt.ylabel('$\mathrm{\hat{\\psi}(U^T\hat{\\rho})}$')
    # ax.scatter(u4[:-1], alpha_pre[:-1], s=6)   # scatter plot of the predicted B-spline
    # ax.plot(u4[:-1], alpha_pre[:-1], color=(0.7, 0.7, 0.7))
    # ax.plot(u5[:-1], mean_alpha_pre[:-1], '--b')
    # ax.plot(u6[:-1], alpha_function(u6)[:-1], '-r')
    ax.plot(u4, alpha_pre, color=(0.7, 0.7, 0.7))
    ax.plot(u5, mean_alpha_pre, '--b')
    ax.plot(u6, alpha_function(u6), '-r')
lines, = plt.plot([], [], '-r', label='Real curve')
lines2, = plt.plot([], [], '--b', label='Single index')
plt.legend(loc='upper left', bbox_to_anchor=(0.15,1), frameon=False)
plt.show()


# ------------------------------------- Save data and figures -----------------------------------------
folder_name = os.path.join(os.path.dirname(__file__), 'results', f'model2,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}')
os.makedirs(folder_name, exist_ok=True)

# save figures
fig1_path = os.path.join(folder_name, f'model2_sim_mean,m={m},n={n},nodenum1={nodenum1},mean_censor_rate={mean_censor_rate:.3f}.png')
fig1.savefig(fig1_path, dpi=600)
fig2_path = os.path.join(folder_name, f'model2_sim_shape,m={m},n={n},nodenum2={nodenum2},mean_censor_rate={mean_censor_rate:.3f}.png')
fig2.savefig(fig2_path, dpi=600)

# save variables
var_path = os.path.join(folder_name, f'model2_sim_variables,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}.xlsx')
var_data = np.hstack([z,w,x,u, g_true_m,g_mean_m,alpha_true_m,alpha_mean_m])
var_df1 = pd.DataFrame(var_data, columns=['z1','z2','w1','w2','x1','x2','u1','u2','g_true','g_mean','alpha_true','alpha_mean'])
var_df2 = pd.DataFrame(g_pre_m)
var_df3 = pd.DataFrame(alpha_pre_m)
with pd.ExcelWriter(var_path, engine='xlsxwriter') as writer:
    var_df1.to_excel(writer, index=False, sheet_name='model2_variables')
    var_df2.to_excel(writer, index=False, sheet_name='g_pre')
    var_df3.to_excel(writer, index=False, sheet_name='alpha_pre')

# save data
data_path = os.path.join(folder_name, f'model2_sim_results,m={m},n={n},mean_censor_rate={mean_censor_rate:.4f}.xlsx')
df1 = pd.DataFrame({'sample_size_n':[n]+[None]*(m-1), 'num_reps_m':[m]+[None]*(m-1),'mean_single_index':[g_func]+[None]*(m-1),'shape_single_index':[alpha_func]+[None]*(m-1),
        'nodenum1':[nodenum1]+[None]*(m-1), 'degree1':[degree1]+[None]*(m-1),
        'nodenum2':[nodenum2]+[None]*(m-1), 'degree2':[degree2]+[None]*(m-1),
        'censor_value':[censor_value]+[None]*(m-1), 'censor_rate':censor_rate_m[:,0],
        'mean_censor_rate':[mean_censor_rate]+[None]*(m-1)
        })

df2_data = np.hstack([gamma_m, beta_m, s_m, rho_m, eta_m, zeta_m])
df2 = pd.DataFrame(df2_data, columns=['γ1','γ2','β1','β2','s1','s2','ρ1','ρ2',
                                      'η1','η2','η3','η4','η5','η6',
                                      'ζ1','ζ2','ζ3','ζ4','ζ5','ζ6'])

max_len = max(len(gamma_true),len(beta_true),len(eta_true),len(zeta_true))
gamma_len = max_len-len(gamma_true)
beta_len = max_len-len(beta_true)
s_len = max_len-len(s_true)
rho_len = max_len-len(rho_true)
eta_len = max_len-len(eta_true)
zeta_len = max_len-len(zeta_true)
df3 = pd.DataFrame({'γ_95low':list(gamma_95low)+[None]*gamma_len, 'γ_95upper':list(gamma_95upper)+[None]*gamma_len,'γ_true':list(gamma_true)+[None]*gamma_len, 'γ_mean':list(gamma_mean)+[None]*gamma_len, 'γ_bias':list(gamma_bias)+[None]*gamma_len, 'γ_sd':list(gamma_sd)+[None]*gamma_len,'γ_mse':list(gamma_mse)+[None]*gamma_len,
                    'β_95low':list(beta_95low)+[None]*beta_len,   'β_95upper':list(beta_95upper)+[None]*beta_len,  'β_true':list(beta_true)+[None]*beta_len,   'β_mean':list(beta_mean)+[None]*beta_len,   'β_bias':list(beta_bias)+[None]*beta_len,   'β_sd':list(beta_sd)+[None]*beta_len,  'β_mse':list(beta_mse)+[None]*beta_len,
                    's_95low':list(s_95low)+[None]*s_len,         's_95upper':list(s_95upper)+[None]*s_len,        's_true':list(s_true)+[None]*s_len,         's_mean':list(s_mean)+[None]*s_len,         's_bias':list(s_bias)+[None]*s_len,         's_sd':list(s_sd)+[None]*s_len,        's_mse':list(s_mse)+[None]*s_len,
                    'ρ_95low':list(rho_95low)+[None]*rho_len,     'ρ_95upper':list(rho_95upper)+[None]*rho_len,    'ρ_true':list(rho_true)+[None]*rho_len ,    'ρ_mean':list(rho_mean)+[None]*rho_len,     'ρ_bias':list(rho_bias)+[None]*rho_len,     'ρ_sd':list(rho_sd)+[None]*rho_len,    'ρ_mse':list(rho_mse)+[None]*rho_len,
                    'η_95low':list(eta_95low)+[None]*eta_len,     'η_95upper':list(eta_95upper)+[None]*eta_len,    'η_true':list(eta_true)+[None]*eta_len,     'η_mean':list(eta_mean)+[None]*eta_len,     'η_bias':list(eta_bias)+[None]*eta_len,     'η_sd':list(eta_sd)+[None]*eta_len,    'η_mse':list(eta_mse)+[None]*eta_len,
                    'ζ_95low':list(zeta_95low)+[None]*zeta_len,   'ζ_95upper':list(zeta_95upper)+[None]*zeta_len,  'ζ_true':list(zeta_true)+[None]*zeta_len,   'ζ_mean':list(zeta_mean)+[None]*zeta_len,   'ζ_bias':list(zeta_bias)+[None]*zeta_len,   'ζ_sd':list(zeta_sd)+[None]*zeta_len,  'ζ_mse':list(zeta_mse)+[None]*zeta_len,
                    'g_func_IABIAS':[g_func_IABIAS]+[None]*(max_len-1), 'g_func_MISE':[g_func_MISE]+[None]*(max_len-1),
                    'alpha_func_IABIAS':[alpha_func_IABIAS]+[None]*(max_len-1), 'alpha_func_MISE':[alpha_func_MISE]+[None]*(max_len-1)
                    })

with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
    df1.to_excel(writer, index=False, sheet_name='model2_set')
    df2.to_excel(writer, index=False, sheet_name='estimates')
    df3.to_excel(writer, index=False, sheet_name='evaluation_metrics')


# ---------------------------------------- Compute the running time -----------------------------------------
end_time = time.time()   # record the end time of the program
run_time = end_time - start_time   # compute the running time
print("Running time:", int(run_time // 60), "min", int(run_time % 60), "s")
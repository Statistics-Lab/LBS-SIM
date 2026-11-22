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
import scienceplots
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
    base_function_dao = []
    for x_i in x:
        base_function_dao.append([bs.base_b_spline_dao(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
    return np.array(base_function_dao)  # n x (m+k-1)

# define the second-derivative matrix of the basis functions for a 1-D array
def base_function_dao2(x):
    bs = BS_curve(nodenum, degree, x, min(x), max(x))
    base_function_dao2 = []
    for x_i in x:
        base_function_dao2.append([bs.base_b_spline_dao2(i, bs.k, x_i) for i in range(0, bs.nodenum + bs.k - 1)])
    return np.array(base_function_dao2)  # n x (m+k-1)

# ------------------------------------- Define initial values for the real-data example --------------------------------
def shili_chuzhi(z,w,y):
    # ------- least-squares estimation of gamma and beta ---------
    X = np.hstack([z,w]) # design matrix
    A = np.linalg.lstsq(X, y, rcond=None)[0]
    gamma = A[:z_num]
    beta = A[z_num:]

    # -------------- normalize beta (divide by its norm), and obtain gamma, eta as initial values ---------------
    beta = beta/np.linalg.norm(beta)  # initial value of beta
    lambd = beta[1:]/beta[0]

    X = np.hstack([z,base_function(w@beta)])
    A = np.linalg.lstsq(X, y, rcond=None)[0]
    gamma = A[:z_num]
    eta = A[z_num:]
    print('Initial parameters when solving for initial values:','gamma=',gamma,'beta=',beta,'eta=',eta)
    # -------------- iterative solution --------------
    iter_chuzhi = 0
    while iter_chuzhi < m:
        for i in range(100):
            theta = np.hstack((gamma.T, lambd.T, eta.T)).T  # initial vector
            print('theta=', theta)
            # estimate beta
            Newton_max_iter = 300  # set the maximum Newton iterations to avoid an infinite loop
            for j in range(Newton_max_iter):
                lambd = beta[1:]/beta[0]
                lambd_1 = np.insert(lambd,0,1)  # a 1 is prepended to lambd
                betalambd = -np.outer(lambd,lambd_1)*(np.sum(lambd_1**2)**(-3/2)) + np.eye(w_num)[1:]*(np.sum(lambd_1**2)**(-1/2))
                betalambd2 = -np.kron(lambd.reshape(w_num-1,1),np.eye(w_num)[1:])*(np.sum(lambd_1**2)**(-3/2)) -np.kron(np.eye(w_num-1).flatten().reshape((w_num-1)**2,1),lambd_1)*(np.sum(lambd_1**2)**(-3/2)) \
                             +np.kron(lambd.reshape(w_num-1,1),np.outer(lambd,lambd_1))*3*(np.sum(lambd_1**2)**(-5/2)) -np.kron(np.eye(w_num)[1:],lambd.reshape(w_num-1,1))*(np.sum(lambd_1**2)**(-3/2))
                lambd1 = lambd - np.linalg.inv(2 * betalambd@w.T@np.diag((base_function_dao(w@beta)@eta)**2)@w@betalambd.T \
                                 - 2 * betalambd@w.T@np.diag((y-z@gamma-base_function(w@beta)@eta)*(base_function_dao2(w@beta)@eta))@w@betalambd.T \
                                 - 2 * (((y-z@gamma-base_function(w@beta)@eta)*(base_function_dao(w@beta)@eta))@w@betalambd2.T).reshape(w_num-1,w_num-1) ) \
                              @ (-2*((y-z@gamma-base_function(w@beta)@eta)*(base_function_dao(w@beta)@eta))@w@betalambd.T)
                beta = np.insert(lambd1,0,1) / np.sqrt(1+np.sum(lambd1**2))
                thre = max(abs(lambd1 - lambd))

                if thre < 1e-4:
                    lambd = lambd1
                    break
                else:
                    lambd = lambd1

            # estimate gamma and eta
            X = np.hstack([z, base_function(w @ beta)])
            A = np.linalg.lstsq(X, y, rcond=None)[0]
            gamma = A[:z_num]
            eta = A[z_num:]

            theta1 = np.hstack((gamma.T, lambd.T, eta.T)).T
            thre1 = max(abs(theta1 - theta))
            if thre1 < 1e-4:
                iter_chuzhi += 1
                break
    return theta1


# ---------------------------------------- True parameter values ---------------------------------------
# # number of knots = 5
# alpha_true = np.array([1.02498481])
# gamma_true = np.array([0.34495561,  0.12388933, -0.47600271, -0.55366528, -0.59356164, -0.89816981,  -0.49510901])
# beta_true = np.array([0.97772121, 0.20990769])
# eta_true = np.array([6.5148693,  7.51231329, 8.68667829, 8.20854921, 7.51462087, 9.12025714, 6.96114064])

# number of knots = 4
alpha_true = np.array([1.02399773])
gamma_true = np.array([0.34345628,  0.14684071, -0.46783227, -0.54361971, -0.59682428, -0.90211135, -0.49702163])
beta_true = np.array([0.98546586, 0.16987361])
eta_true = np.array([6.65820057, 7.3786672 , 9.12567311, 7.27684168, 9.03145268, 7.09175597])


# -------------------------------------- Real-data predictors and response -------------------------------
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'GBSG2_data.csv'))
data = np.array(df)
n = data.shape[0]  # sample size
# categorical predictor z
z = data[:,[0,1,2,3,4,5,6]]
z_true = z
z_num = z.shape[1]  # number of categorical variables
print('Categorical predictor z = ',z,z.shape)

# continuous predictor w
# w = data[:,8:10]
w1= (data[:,8]-min(data[:,8]))/(max(data[:,8])-min(data[:,8]))
w2= (data[:,9]-min(data[:,9]))/(max(data[:,9])-min(data[:,9]))
w = np.c_[w1,w2]
w_true= w
w_num = w.shape[1] # number of continuous variables
print('Continuous predictor w = ',w,w.shape)

# response variable y
y_true = log(data[:,10])  # n x 1, log-transformed time
print('Response variable y_true = ', y_true.shape, '\ny_true range: [{},{}]'.format(min(y_true),max(y_true)))
# censoring rate
ind_true = data[:,11]  # 0 = censored, 1 = event
censor_rate_true = (n - np.sum(ind_true, axis=0)) / n
print("Censoring rate:", censor_rate_true)

# ---------------------------------------- Iterative estimation ----------------------------------------

m = 1000    # total number of outer-loop iterations
iter = 0  # number of iterations completed so far
nodenum = 4   # number of B-spline knots
degree = 3   # degree of the B-spline

# storage arrays: hold each estimated value
gamma_m = np.zeros((m, z_num))
beta_m = np.zeros((m, w_num))
alpha_m = np.zeros((m,1))
eta_m = np.zeros((m,nodenum + degree - 1))
censor_rate_m = np.zeros((m,1))
g_func_m = np.zeros((m,n))


while iter < m:
    print('Iteration', iter+1, ':')

    # generate the response y
    N = np.random.normal(0, 1, n)

    epsilon = 2 * np.arcsinh(alpha_true * N * 0.5)  # n x 1
    print('\nerror term epsilon range: [{},{}]'.format(min(epsilon), max(epsilon)))
    y = z_true @ gamma_true + base_function(w_true @ beta_true) @ eta_true + epsilon  # n x 1


    # censoring rate
    # ind = data[:, 11]  # 0 = censored, 1 = event
    # ind_replace = np.random.choice(ind, size=n-np.sum(ind), replace=True)
    # ind[ind == 0] = ind_replace

    ind = np.ones((n,1)).flatten()
    censor_rate = (n - np.sum(ind, axis=0)) / n
    print("Censoring rate:", censor_rate)
    censor_rate_m[iter] = censor_rate

# ----------------------------------- Maximum likelihood estimation -- Newton-Raphson ------------------------------
    gamma = gamma_true
    beta = beta_true
    lambd = beta[1:]/beta[0]
    eta = eta_true
    alpha = alpha_true

    # initial-value vector
    theta0 = np.hstack((gamma.T,beta.T,alpha.T,eta.T)).T
    print('Initial theta0 = ',theta0, theta0.shape)
    print('Initial gamma = ', gamma, gamma.shape, '\ninitial beta = ', beta, beta.shape, '\ninitial alpha =',alpha, alpha.shape, '\ninitial eta = ', eta, eta.shape)

    Newton_max_iter = 100  # set the maximum Newton iterations to avoid an infinite loop
    for j in range(Newton_max_iter):
        print('Iteration:', j)
        if bool(alpha>2) is True:
            break

        # reparameterize beta via lambda
        lambd = beta[1:]/beta[0]
        lambd_1 = np.insert(lambd,0,1)   # a 1 is prepended to lambd
        betalambd = -np.outer(lambd,lambd_1)*(np.sum(lambd_1**2)**(-3/2)) + np.eye(w_num)[1:]*(np.sum(lambd_1**2)**(-1/2))
        betalambd2 = -np.kron(lambd.reshape(w_num-1,1),np.eye(w_num)[1:])*(np.sum(lambd_1**2)**(-3/2)) -np.kron(np.eye(w_num-1).flatten().reshape((w_num-1)**2,1),lambd_1)*(np.sum(lambd_1**2)**(-3/2)) \
                     +np.kron(lambd.reshape(w_num-1,1),np.outer(lambd,lambd_1))*3*(np.sum(lambd_1**2)**(-5/2)) -np.kron(np.eye(w_num)[1:],lambd.reshape(w_num-1,1))*(np.sum(lambd_1**2)**(-3/2))

        theta = np.hstack((gamma.T, lambd.T, alpha.T, eta.T)).T
        print('theta at iteration start = ', theta, theta.shape)

        z_gamma = z @ gamma
        w_beta = w @ beta  # n x 1

        # common factors before taking derivatives
        mu = z @ gamma + base_function(w_beta) @ eta  # n x 1

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
        print( '\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))

        # if d contains NaN or inf, break the Newton iterations
        if np.isnan(d_dao).any() or np.isinf(d_dao).any():
            break

        # first derivatives
        lgamma = z.T @ (ind*(-0.5*tanh2+(1/(alpha**2))*sinh1)) + z.T @ ((1-ind)*(d*(1/alpha)*cosh2))
        llambd = (betalambd @ w.T) @ (ind * ((-0.5*tanh2+(1/(alpha**2))*sinh1))*(base_function_dao(w_beta) @ eta))  + (betalambd @ w.T) @ ((1-ind) * (d*(1/alpha)*cosh2)*(base_function_dao(w_beta) @ eta))
        lalpha = (ind @ (((4/(alpha**2))*(sinh2**2))-1))*(1/alpha) + ((1-ind) @ (d*sinh2)) *(2/(alpha**2))
        leta = base_function(w_beta).T @ (ind * (-0.5*tanh2 + (1/(alpha**2))*sinh1 ))  + base_function(w_beta).T @ ((1-ind) * (d*(1/alpha)*cosh2))  # (m+k-1) x 1

        # second derivatives
        lgamma2 = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ z + z.T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ z
        lgammalambd = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1)))*(base_function_dao(w_beta) @ eta)) @ w @ betalambd.T + z.T @ np.diag((1-ind)*(((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))*(base_function_dao(w_beta) @ eta)) @ w @ betalambd.T #3×4
        lgammaalpha = z.T @ (ind * ((-2/(alpha**3))*sinh1)) + z.T @ ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d)))
        lgammaeta = z.T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ base_function(w_beta) + z.T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ base_function(w_beta)   #3×(m+k-1)
        llambd2 = betalambd @ w.T @ np.diag(ind * ((((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))*((base_function_dao(w_beta) @ eta)**2))) @ w @ betalambd.T \
                  + betalambd @ w.T @ np.diag(ind * ((-0.5*tanh2 + (1/(alpha**2))*sinh1 )*(base_function_dao2(w_beta) @ eta))) @ w @ betalambd.T \
                  + ((betalambd2 @ w.T) @ (ind * (-0.5*tanh2 + (1/(alpha**2))*sinh1 )*(base_function_dao(w_beta) @ eta))).reshape(w_num-1,w_num-1) \
                  + betalambd @ w.T @ np.diag( (1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))*((base_function_dao(w_beta) @ eta)**2) ) @ w @ betalambd.T \
                  + betalambd @ w.T @ np.diag((1-ind) * ((1/alpha)*cosh2*d ) * (base_function_dao2(w_beta) @ eta)) @ w @ betalambd.T \
                  + ((betalambd2 @ w.T) @ ((1-ind) * ((1/alpha)*cosh2*d ) * (base_function_dao(w_beta) @ eta))).reshape(w_num-1,w_num-1)  #4×4
        llambdalpha = betalambd @ w.T @ (ind * ((-2/(alpha**3))*sinh1) *(base_function_dao(w_beta) @ eta)) + betalambd @ w.T @ ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d)) *(base_function_dao(w_beta) @ eta))  #4×1
        llambdeta = betalambd @ w.T @ np.diag(ind * ((((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )) )*(base_function_dao(w_beta) @ eta))) @ base_function(w_beta) + betalambd @ w.T @ np.diag(ind*(-0.5*tanh2 + (1/(alpha**2))*sinh1)) @ base_function_dao(w_beta)  \
                    +betalambd @ w.T @ np.diag((1-ind) * ((((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) *(base_function_dao(w_beta) @ eta)) @ base_function(w_beta) + betalambd @ w.T @ np.diag((1-ind)*((1/alpha)*cosh2*d)) @ base_function_dao(w_beta)    #4×(m+k-1)
        lalpha2 = ind @ ((-12/(alpha**4))*(sinh2**2)+(1/(alpha**2))) + (1-ind) @ ((-d*(4/(alpha**3))*sinh2) - ((-4/(alpha**4))*(sinh2**2)*d_dao  ))  # 1 x 1
        lalphaeta = (ind * ((-2/(alpha**3))*sinh1)).T @ base_function(w_beta) + ((1-ind) * (((-1/(alpha**3))*sinh1*d_dao) - ((1/(alpha**2))*cosh2*d))).T @ base_function(w_beta)   # 1 x (m+k-1)
        leta2 = base_function(w_beta).T @ np.diag(ind * (((0.25/(cosh2**2)) - ((1/(alpha**2))*cosh1 )))) @ base_function(w_beta) + base_function(w_beta).T @ np.diag((1-ind) * (((-1/(alpha**2))*d_dao*(cosh2**2) ) - ((-0.5/alpha)*sinh2*d ))) @ base_function(w_beta)  # (m+k-1) x (m+k-1)

        ltheta = np.hstack((lgamma.T,llambd.T,lalpha.T,leta.T)).T
        print('ltheta=',ltheta,max(abs(ltheta)))
        ltheta2 = np.block([[lgamma2                    ,lgammalambd                ,lgammaalpha.reshape(z_num,1) ,lgammaeta],
                            [lgammalambd.T              ,llambd2                    ,llambdalpha.reshape(w_num-1,1) ,llambdeta],
                            [lgammaalpha.T.reshape(1,z_num) ,llambdalpha.T.reshape(1,w_num-1) ,lalpha2                  ,lalphaeta.reshape(1,nodenum+degree-1)],
                            [lgammaeta.T                ,llambdeta.T                ,lalphaeta.T.reshape(nodenum+degree-1,1) ,leta2]])   #(3+4+1+m+k-1)×(3+4+1+m+k-1)

        theta1 = theta - np.linalg.inv(ltheta2) @ ltheta
        print('theta1= ',theta1, theta1.shape)
        # eig_values = np.linalg.eigvals(np.linalg.inv(ltheta2))   # eigenvalues of the inverse Hessian
        # is_negative_matrix = all([eig<0 for eig in eig_values])
        # print('negative definite?', is_negative_matrix)

        # SE = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
        # print('SE=',SE)

        gamma = theta1[0:z_num]
        lambd = theta1[z_num:z_num+w_num-1]
        beta = np.insert(lambd,0,1) / np.sqrt(1+np.sum(lambd**2))
        alpha = theta1[z_num+w_num-1:z_num+w_num]
        eta = theta1[z_num+w_num:]

        likelihood_value = (ind@(log(1/sqrt(2*pi))+log(1/alpha)+log(np.cosh((y-(z@gamma+base_function(w@beta)@eta))/2))\
                            -(2/(alpha**2))*np.sinh((y-(z@gamma+base_function(w@beta)@eta))/2)**2)\
                            +(1-ind)@(log(1-standardN.cdf((2/alpha)*np.sinh((y-(z@gamma+base_function(w@beta)@eta))/2)))) )
        print('likelihood_value =', likelihood_value)
        AIC = -2*likelihood_value+2*len(theta1)
        print('AIC =', AIC)

        thre = max(abs(theta1 - theta))
        print('Step size:', thre)
        if (thre > 1e+3) or (j > 60 and thre > 0.05) or (j == Newton_max_iter-1):
            break
        if thre < 1e-5:
            gamma_m[iter] = gamma
            beta_m[iter] = beta
            alpha_m[iter] = alpha
            eta_m[iter] = eta
            print('theta1 at convergence = ', np.hstack((theta1[0:z_num].T, beta.T, theta1[z_num+w_num-1:z_num+w_num].T, theta1[z_num+w_num:].T)).T)

            mark = 1
            iter += 1
            break
        else:
            print('theta1 when not converged = ', np.hstack((gamma.T, beta.T, alpha.T, eta.T)).T )



print('gamma_m = ', gamma_m, gamma_m.shape)
print('beta_m = ', beta_m, beta_m.shape)
print('alpha_m = ', alpha_m, alpha_m.shape)
print('eta_m = ', eta_m, eta_m.shape)


# -------------------------- 95% confidence intervals of the parameters -----------------------------------
gamma_95low = np.percentile(gamma_m, 2.5, axis=0)
gamma_95upper = np.percentile(gamma_m, 97.5, axis=0)
beta_95low = np.percentile(beta_m, 2.5, axis=0)
beta_95upper = np.percentile(beta_m, 97.5, axis=0)
alpha_95low = np.percentile(alpha_m, 2.5, axis=0)
alpha_95upper = np.percentile(alpha_m, 97.5, axis=0)



# ---------------------------------------- Plotting -----------------------------------------
# plt.style.use(['science','vibrant'])

plt.rcParams['axes.unicode_minus'] = False   # avoid the minus-sign rendering issue when saving figures
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.figure(dpi=600)


fig1, ax = plt.subplots()
for fig_iter in range(m):
    u1 = np.sort(w @ beta_m[fig_iter].T).flatten()
    g_pre = base_function(u1) @ eta_m[fig_iter].T
    u2 = np.sort(w_true @ beta_true).flatten()
    g_true = base_function(u2) @ eta_true

    ax.set_title(f"model1, n={n}, censor_rate={censor_rate_true:.3f}")
    plt.xlabel('$\mathrm{W^T \hat{\\beta}}$')
    plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')
    ax.plot(u1[:-1], g_pre[:-1], '--b')  # plot of the estimated mean
    ax.plot(u2[:-1], g_true[:-1], '-r')
plt.show()

fig1_95, ax = plt.subplots()
for fig_iter in range(m):
    u1 = np.sort(w @ beta_m[fig_iter].T).flatten()
    g_func_m[fig_iter] = base_function(u1) @ eta_m[fig_iter].T
u2 = np.sort(w_true @ beta_true).flatten()
g_true = base_function(u2) @ eta_true
g_pre_low = np.percentile(g_func_m, 2.5, axis=0)
g_pre_upper = np.percentile(g_func_m, 97.5, axis=0)

ax.set_title(f"model1, n={n}, censor_rate={censor_rate_true:.3f}")
plt.xlabel('$\mathrm{W^T \hat{\\beta}}$')
plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')
ax.plot(u2[:-1], g_true[:-1], '-r')
ax.plot(u2[:-1], g_pre_low[:-1], '--r')
ax.plot(u2[:-1], g_pre_upper[:-1], '--r')

lines, = plt.plot([], [], '-r', label='Estimated curve')
lines2, = plt.plot([], [], '--r', label='95% C.I')
# plt.legend(loc='upper left', frameon=False)
plt.legend()
plt.show()


# ------------------------------------- Save data and figures -----------------------------------------
folder_name = os.path.join(os.path.dirname(__file__), 'results', f'model1,m={m},n={n},censor_rate={censor_rate:.4f}')
os.makedirs(folder_name, exist_ok=True)

# save figures
fig_path = os.path.join(folder_name, f'application_model1_single_index,m={m},n={n},nodenum={nodenum},censor_rate={censor_rate:.3f}.png')
fig1.savefig(fig_path, dpi=600)
fig_path = os.path.join(folder_name, f'application_model1_CI_band,m={m},n={n},nodenum={nodenum},censor_rate={censor_rate:.3f}.png')
fig1_95.savefig(fig_path, dpi=600)

# save results
data_path = os.path.join(folder_name, f'application_model1_bootstrap_CI_results,m={m},n={n},censor_rate={censor_rate:.4f}.xlsx')
df1 = pd.DataFrame(g_func_m.reshape(n,m))
data = np.hstack([g_true.reshape(n,1),g_pre_low.reshape(n,1),g_pre_upper.reshape(n,1)])
df2 = pd.DataFrame(data, columns=['g_true','g_pre_low','g_pre_upper'])

df3_data = np.hstack([gamma_m, beta_m, alpha_m, eta_m])
df3 = pd.DataFrame(df3_data, columns=['γ1','γ2','γ3','γ4','γ5','γ6','γ7','β1','β2','α',
                                      'η1','η2','η3','η4','η5','η6'])

max_len = max(len(gamma),len(beta),len(alpha))
gamma_len = max_len-len(gamma)
beta_len = max_len-len(beta)
alpha_len = max_len-len(alpha)
df4 = pd.DataFrame({'γ_95low':list(gamma_95low)+[None]*gamma_len, 'γ_95upper':list(gamma_95upper)+[None]*gamma_len, 'γ_true':list(gamma_true)+[None]*gamma_len,
                    'β_95low':list(beta_95low)+[None]*beta_len,   'β_95upper':list(beta_95upper)+[None]*beta_len,   'β_true':list(beta_true)+[None]*beta_len,
                    'α_95low':list(alpha_95low)+[None]*alpha_len, 'α_95upper':list(alpha_95upper)+[None]*alpha_len, 'α_true':list(alpha_true)+[None]*alpha_len
                    })

with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
    df1.to_excel(writer, index=False, sheet_name='single_index_each_iter')
    df2.to_excel(writer, index=False, sheet_name='95pct_CI_band')
    df3.to_excel(writer, index=False, sheet_name='estimates')
    df4.to_excel(writer, index=False, sheet_name='95pct_param_CI')


# ---------------------------------------- Compute the running time -----------------------------------------
end_time = time.time()  # record the end time of the program
run_time = end_time - start_time  # compute the running time
print("Running time:", int(run_time // 60), "min", run_time % 60, "s")
import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib as mpl
import matplotlib.pyplot as plt   #version 3.7.2
from pylab import *
import scipy as sp
from scipy.stats import norm
from scipy import special
from scipy import stats
from sklearn.utils import resample
from scipy.linalg import sqrtm
import seaborn as sns
import time

start_time = time.time()  # record the start time of the program

#---------------------------------------- Define B-spline ----------------------------------------
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
# # number of knots = 5 (3 variables)
# gamma_true = np.array([0.37572247,  0.19264368, -0.19652263, -0.23318559, -0.5865165,  -0.97250622, -0.51623045])
# beta_true = np.array([0.99459824, 0.10379948])
# lambd_true = beta_true[1:] / beta_true[0]
# eta_true = np.array([5.5919241,  7.39739896 ,9.09097222, 7.68068838, 7.45741125, 8.18034078, 7.38627857])
# s_true = np.array([ 0.27029412,  0.43520438, -0.3566604])
# rho_true = np.array([ 0.99936559, -0.03561492])
# tau_true = rho_true[1:] / rho_true[0]
# zeta_true = np.array([-3.63579222 , 0.96675163 , 0.01629673, -0.26013047 ,-0.46798762,  0.26692969, -1.53231759])

# # number of knots = 4, 4 (3 variables)
# gamma_true = np.array([0.32534996, 0.22912254, -0.31988332, -0.37776451, -0.58711186, -0.95044111, -0.49597332])
# beta_true = np.array([0.98503096, 0.17237753])
# lambd_true = beta_true[1:] / beta_true[0]
# eta_true = np.array([6.50072244, 7.22129965, 9.0976504, 6.99757992, 8.79013548, 7.06273947])
# s_true = np.array([0.19228124, 0.37428031, -0.38845153])
# rho_true = np.array([0.05901393, 0.99825716])
# tau_true = rho_true[1:] / rho_true[0]
# zeta_true = np.array([1.30619112, -0.62340405, -0.04405613, -0.36876146, 0.51209478, 0.04205802])

# # number of knots = 4, 3 (3 variables)
# gamma_true = np.array([0.373805, 0.20151641, -0.20884325, -0.15973821, -0.61554332, -0.95448218, -0.50523126])
# beta_true = np.array([0.98124889, 0.19274497])
# lambd_true = beta_true[1:] / beta_true[0]
# eta_true = np.array([5.59087296, 7.69580591, 9.03014042, 6.75968839, 8.89897141,   7.00542592])
# s_true = np.array([0.31862577,  0.52492195, -0.31168809])
# rho_true = np.array([0.81537162, 0.57893793])
# tau_true = rho_true[1:] / rho_true[0]
# zeta_true = np.array([-3.71357726,  1.45083239, -1.74665585,  0.61589707, -1.1686208])

# number of knots = 4, 3 (7 variables)
gamma_true = np.array([0.39788857, 0.19615306, -0.21721457, -0.15755339, -0.614013, -0.95053472, -0.49967816])
beta_true = np.array([0.98155832, 0.19116293])
lambd_true = beta_true[1:] / beta_true[0]
eta_true = np.array([5.60143837, 7.67248416, 9.04782099, 6.73497435, 8.94157181, 6.98026004])
s_true = np.array([0.04075597, -0.02059978, 0.31339343, 0.52518223, 0.00144632, -0.31027257, 0.00383746])
rho_true = np.array([0.80982866, 0.58666647])
tau_true = rho_true[1:] / rho_true[0]
zeta_true = np.array([-3.70536797, 1.43405329, -1.75106489, 0.63385971, -1.14142091])


#---------------------------------------- Real-data variables ------------------------------------------------

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'GBSG2_data.csv'))
data = np.array(df)
n = data.shape[0]  # sample size
# categorical predictor z
z = data[:,[0,1,2,3,4,5,6]]
z_num = z.shape[1]  # number of categorical variables
print('Categorical predictor z = ',z,z.shape)

# continuous predictor w
# w = data[:,8:10]
w1= (data[:,8]-min(data[:,8]))/(max(data[:,8])-min(data[:,8]))
w2= (data[:,9]-min(data[:,9]))/(max(data[:,9])-min(data[:,9]))
w = np.c_[w1,w2]
w_true = w
w_num = w.shape[1]  # number of continuous variables
print('Continuous predictor w = ',w,w.shape)

# categorical predictor x
x = data[:,[0,1,2,3,4,5,6]]
x_num = x.shape[1]  # number of categorical variables
print('Categorical predictor x = ',x,x.shape)

# continuous predictor u
# u = data[:,8:10]
u1= (data[:,8]-min(data[:,8]))/(max(data[:,8])-min(data[:,8]))
u2= (data[:,9]-min(data[:,9]))/(max(data[:,9])-min(data[:,9]))
u = np.c_[u1,u2]
u_true = u
u_num = u.shape[1]  # number of continuous variables
print('Continuous predictor u = ',u,u.shape)

# response variable y
y = log(data[:,10])  # n x 1, log-transformed time
print('Response variable y = ', y, y.shape, '\ny range: [{},{}]'.format(min(y),max(y)))

# censoring rate
ind = data[:,11]  # 0 = censored, 1 = event
censor_rate = (n - np.sum(ind, axis=0)) / n
print("Censoring rate:", censor_rate)


#--------------------------------------- Iterative estimation --------------------------------------------
m = 1000   # total number of outer-loop iterations
iter = 0   # number of iterations completed so far
nodenum1 = 4   # number of B-spline knots
degree1 = 3   # degree of the B-spline
nodenum2 = 3
degree2 = 3

# storage arrays: hold each estimated value; initial values are not stored
gamma_m = np.zeros((m,z_num))
beta_m = np.zeros((m,w_num))
s_m = np.zeros((m,x_num))
rho_m = np.zeros((m,u_num))
eta_m = np.zeros((m,nodenum1 + degree1 - 1))
zeta_m = np.zeros((m,nodenum2 + degree2 - 1))
censor_rate_m = np.zeros((m,1))
censor_rate_m[iter] = censor_rate
g_func_m = np.zeros((m,n))
alpha_func_m = np.zeros((m,n))


# mark = 0
# for a in np.arange(0, 2, 0.05):   # value of the shape parameter alpha
#     print('a=',a)
#     if mark == 1:
#         break
#     error = 0

mark = 0
while iter < m:
    error = 0
    print('Iteration', iter+1, ':')

    N = np.random.normal(0,1,n)
    z_gamma = z @ gamma_true
    w_beta = w @ beta_true
    u_rho = u @ rho_true
    epsilon = 2*np.arcsinh(np.exp(x@s_true + base_function(u_rho)@zeta_true) * N * 0.5)  # n x 1
    print('\nerror term epsilon range: [{},{}]'.format(min(epsilon),max(epsilon)))

    # generate the response y
    y = z @ gamma_true + base_function(w_beta)@eta_true + epsilon   # n x 1
    print('\nresponse variable y range: [{},{}]'.format(min(y),max(y)))   # keep y from becoming too large

    # censoring rate
    # ind = data[:,11]  # 0 = censored, 1 = event
    ind = np.ones((n,1)).flatten()
    censor_rate = (n - np.sum(ind, axis=0)) / n
    print("Censoring rate:", censor_rate)
    censor_rate_m[iter] = censor_rate


    gamma = gamma_true
    beta = beta_true
    lambd = beta_true[1:] / beta_true[0]
    eta = eta_true
    s = s_true
    rho = rho_true
    tau = rho_true[1:] / rho_true[0]
    zeta = zeta_true


    # initial-value vector
    theta0 = np.hstack((gamma.T, beta.T, s.T, rho.T, eta.T, zeta.T)).T
    print('Initial theta0 = ',theta0, theta0.shape)
    print('Initial gamma = ', gamma, gamma.shape, '\ninitial beta = ', beta, beta.shape, '\ninitial s = ', s, s.shape, '\ninitial rho =',rho, rho.shape, '\ninitial eta = ', eta, eta.shape, '\nzinitial eta = ', zeta, zeta.shape)
# --------------------------------------- Real-data initial values done --------------------------------------

    zhengti_max_iter = 100
    for zhengti in range(zhengti_max_iter):
        if error == 1 or mark == 1:
            break
        Theta = np.hstack(( gamma.T, lambd.T, s.T, tau.T, eta.T, zeta.T)).T

# ----------------------------------- Maximum likelihood estimation -- Newton iterations ------------------------------------
    # --------------------------------- Fix the mean part; Newton iterations for the shape parameters ---------------------------------
        Newton_max_iter = 200  # set the maximum Newton iterations to avoid an infinite loop
        for j in range(Newton_max_iter):
            print('Shape parameter, iteration:', j)

            tau = rho[1:]/rho[0]
            tau_1 = np.insert(tau,0,1)  #a 1 is prepended to tau
            rhotau = -np.outer(tau,tau_1)*(np.sum(tau_1**2)**(-3/2)) + np.eye(u_num)[1:]*(np.sum(tau_1**2)**(-1/2))
            rhotau2 = -np.kron(tau.reshape(u_num-1,1),np.eye(u_num)[1:])*(np.sum(tau_1**2)**(-3/2)) -np.kron(np.eye(u_num-1).flatten().reshape((u_num-1)**2,1),tau_1)*(np.sum(tau_1**2)**(-3/2)) \
                         +np.kron(tau.reshape(u_num-1,1),np.outer(tau,tau_1))*3*(np.sum(tau_1**2)**(-5/2)) -np.kron(np.eye(u_num)[1:],tau.reshape(u_num-1,1))*(np.sum(tau_1**2)**(-3/2))

            theta = np.hstack(( s.T, tau.T, zeta.T)).T   ## add parameter estimates for the categorical variables
            # print('Shape-parameter theta =', theta, theta.shape)

            w_beta = w @ beta  # n x 1
            u_rho = u @ rho  # n x 1

            # common factors before taking derivatives
            v = np.exp( x @ s + base_function(u_rho) @ zeta)  # n x 1
            # print('\nv range: [{},{}]'.format(min(v),max(v)))
            print('Number exceeding 2:', np.sum(v > 2))
            # if any(list(num <= 0 or num >= 2 for num in v)): # the generator expression is a Python generator object and needs conversion via list
            #     error = 1
            #     break

            mu = z @ gamma + base_function(w_beta) @ eta  # n x 1
            sinh1 = np.sinh(y - mu)
            cosh1 = np.cosh(y - mu)
            sinh2 = np.sinh((y - mu)/2)
            cosh2 = np.cosh((y - mu)/2)
            tanh2 = np.tanh((y - mu)/2)
            standardN = stats.norm(loc=0, scale=1)

            logd = np.log(standardN.pdf((2/v)*sinh2))-np.log(1-standardN.cdf((2/v)*sinh2))
            d = np.exp(logd)
            # print('\nd range: [{},{}]'.format(min(d),max(d)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d).any() or np.isinf(d).any():
                error = 1
                break

            logd_dao = np.log((-(2/v)*sinh2*standardN.pdf((2/v)*sinh2))*(1-standardN.cdf((2/v)*sinh2))+(standardN.pdf((2/v)*sinh2))**2)-np.log((1-standardN.cdf((2/v)*sinh2))**2)
            d_dao = np.exp(logd_dao)
            # print('\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))
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


            ltheta = np.hstack((ls.T, ltau.T, lzeta.T)).T
            # print('ltheta =', ltheta, max(abs(ltheta)))
            if np.isnan(ltheta).any() or np.isinf(ltheta).any():
                error = 1
                break

            ltheta2 = np.block([[ls2,      lstau,       lszeta],
                               [lstau.T,   ltau2,       ltauzeta],
                               [lszeta.T,  ltauzeta.T,  lzeta2]])
            # print('np.diag(ltheta2)=',np.diag(ltheta2))

            theta1 = theta - (np.linalg.inv(ltheta2) @ ltheta)  #(z_num + (w_num-1) + x_num + (x_num-1) + (m1+k1-1) + (m2+k2-1)) × 1
            # print('theta1= ',theta1, theta1.shape)
            eig_values = np.linalg.eigvals(np.linalg.inv(ltheta2))   # eigenvalues of the inverse Hessian
            is_negative_matrix = all([eig < 0 for eig in eig_values])
            # print('negative definite?', is_negative_matrix)

            SE1 = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
            # print('SE1=', SE1)

            s = theta1[0:x_num]
            tau = theta1[x_num:x_num+u_num-1]
            rho = np.insert(tau,0,1) / np.sqrt(1+np.sum(tau**2))
            zeta = theta1[x_num + u_num - 1:]
            thre = max(abs(theta1 - theta))
            print('Shape-parameter step size:', thre)

            if (thre > 1e+3) or (j > 60 and thre > 0.05) or (j == Newton_max_iter - 1):
                error = 1
                break
            elif thre < 1e-4:
                print('Shape-parameter theta1 at convergence = ', np.hstack((s.T, rho.T, zeta.T)).T)
                break
            # else:
            #     print('Shape-parameter theta1 when not converged = ', np.hstack((s.T, rho.T, zeta.T)).T )
        if error == 1:
            break

    # ---------------------------------- Fix the shape-parameter part; Newton iterations for the mean parameters ------------------------------------
        Newton_max_iter = 200  # set the maximum Newton iterations to avoid an infinite loop
        for j in range(Newton_max_iter):
            print('Mean parameter, iteration:', j)
            # reparameterize beta
            lambd = beta[1:]/beta[0]
            lambd_1 = np.insert(lambd,0,1)  # a 1 is prepended to lambd
            betalambd = -np.outer(lambd,lambd_1)*(np.sum(lambd_1**2)**(-3/2)) + np.eye(w_num)[1:]*(np.sum(lambd_1**2)**(-1/2))
            betalambd2 = -np.kron(lambd.reshape(w_num-1,1),np.eye(w_num)[1:])*(np.sum(lambd_1**2)**(-3/2)) -np.kron(np.eye(w_num-1).flatten().reshape((w_num-1)**2,1),lambd_1)*(np.sum(lambd_1**2)**(-3/2)) \
                         +np.kron(lambd.reshape(w_num-1,1),np.outer(lambd,lambd_1))*3*(np.sum(lambd_1**2)**(-5/2)) -np.kron(np.eye(w_num)[1:],lambd.reshape(w_num-1,1))*(np.sum(lambd_1**2)**(-3/2))

            theta = np.hstack((gamma.T, lambd.T, eta.T)).T
            # print('Mean-parameter theta =', theta, theta.shape)

            w_beta = w @ beta  # n x 1
            u_rho = u @ rho  # n x 1

            # common factors before taking derivatives
            v = np.exp( x @ s + base_function(u_rho) @ zeta)  # n x 1
            # print( '\nv range: [{},{}]'.format(min(v),max(v)))
            print('Number exceeding 2:', np.sum(v > 2))
            # if any(list(num <= 0 or num >= 2 for num in v)): # the generator expression is a Python generator object and needs conversion via list
            #     error = 1
            #     break

            mu = z @ gamma + base_function(w_beta) @ eta  # n x 1
            sinh1 = np.sinh(y - mu)
            cosh1 = np.cosh(y - mu)
            sinh2 = np.sinh((y - mu)/2)
            cosh2 = np.cosh((y - mu)/2)
            tanh2 = np.tanh((y - mu)/2)
            standardN = stats.norm(loc=0, scale=1)

            logd = np.log(standardN.pdf((2/v)*sinh2))-np.log(1-standardN.cdf((2/v)*sinh2))
            d = np.exp(logd)
            # print('\nd range: [{},{}]'.format(min(d),max(d)))
            # if d contains NaN or inf, break the Newton iterations
            if np.isnan(d).any() or np.isinf(d).any():
                error = 1
                break

            logd_dao = np.log((-(2/v)*sinh2*standardN.pdf((2/v)*sinh2))*(1-standardN.cdf((2/v)*sinh2))+(standardN.pdf((2/v)*sinh2))**2)-np.log((1-standardN.cdf((2/v)*sinh2))**2)
            d_dao = np.exp(logd_dao)
            # print('\nd_dao range: [{},{}]'.format(min(d_dao),max(d_dao)))
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


            ltheta = np.hstack((lgamma.T, llambd.T, leta.T)).T
            print('ltheta =', ltheta, max(abs(ltheta)))
            if np.isnan(ltheta).any() or np.isinf(ltheta).any():
                error = 1
                break
            ltheta2 = np.block([[lgamma2,          lgammalambd,       lgammaeta],
                                [lgammalambd.T,    llambd2,           llambdeta],
                                [lgammaeta.T,      llambdeta.T,      leta2]])
            # print('np.diag(ltheta2)=', np.diag(ltheta2))

            theta1 = theta - (np.linalg.inv(ltheta2) @ ltheta)  #(z_num + (w_num-1) + x_num + (x_num-1) + (m1+k1-1) + (m2+k2-1)) × 1
            # print('Mean-parameter theta1 = ', theta1, theta1.shape)
            eig_values = np.linalg.eigvals(np.linalg.inv(ltheta2))   # eigenvalues of the inverse Hessian
            is_negative_matrix = all([eig<0 for eig in eig_values])
            # print('negative definite?', is_negative_matrix)

            SE2 = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
            # print('SE2=', SE2)

            gamma = theta1[0:z_num]
            lambd = theta1[z_num:z_num+w_num-1]
            beta = np.insert(lambd,0,1) / np.sqrt(1+np.sum(lambd**2))
            eta = theta1[z_num+w_num-1:]
            thre = max(abs(theta1 - theta))
            print('Mean-parameter step size:', thre)
            if (thre > 1e+3) or (j > 60 and thre > 0.05) or (j == Newton_max_iter - 1):
                error = 1
                break
            if thre < 1e-4:
                print('Mean-parameter theta1 at convergence = ', np.hstack(( gamma.T, beta.T, eta.T)).T)
                break
            # else:
            #     print('Mean-parameter theta1 when not converged = ', np.hstack((gamma.T, beta.T, eta.T)).T )

        if error == 1:
            break

# ---------------------------- Compute standard errors and likelihood ---------------------------------
        # first derivatives
        lgamma = z.T @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1)) + z.T @ ((1-ind) * (d*(1/v)*cosh2))   # z_num x 1
        llambd = (betalambd @ w.T) @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao(w_beta)@eta)) \
               + (betalambd @ w.T) @ ((1-ind) * (d*(1/v)*cosh2) * (base_function_dao(w_beta)@eta))   # (w_num-1) x 1
        ls = x.T @ (ind * ((4/(v**2))*(sinh2**2)-1)) + x.T @  ((1-ind) * (d*sinh2*(2/v)))   # x_num x 1
        ltau = (rhotau @ u.T) @ (ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao(u_rho)@zeta))\
             + (rhotau @ u.T) @ ((1-ind) * (d*sinh2*(2/v)) * (base_function_dao(u_rho)@zeta))   # (u_num-1) x 1
        leta = base_function(w_beta).T @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1))\
             + base_function(w_beta).T @ ((1-ind) * (d*(1/v)*cosh2))   # (m1+k1-1) x 1
        lzeta = base_function(u_rho).T @ (ind * ((4/(v**2))*(sinh2**2)-1))\
              + base_function(u_rho).T @ ((1-ind) * (d*sinh2*(2/v)))   # (m2+k2-1) x 1

        # second derivatives
        lgamma2 = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ z \
                + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ z   # z_num x z_num
        lgammalambd = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * (base_function_dao(w_beta)@eta)) @ w @ betalambd.T \
                    + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * (base_function_dao(w_beta) @ eta)) @ w @ betalambd.T   # z_num x (w_num-1)
        lgammas = z.T @ np.diag(ind * ((-2/(v**2))*sinh1)) @ x \
                + z.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2)) @ x   #z_num × x_num
        lgammatau = z.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(u_rho)@zeta)) @u @rhotau.T \
                  + z.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(u_rho)@zeta)) @u @rhotau.T   #z_num × (u_num-1)
        lgammaeta = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ base_function(w_beta) \
                  + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ base_function(w_beta)   #z_num × (m1+k1-1)
        lgammazeta = z.T @ np.diag(ind * ((-2/(v**2))*sinh1)) @ base_function(u_rho) \
                   + z.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2)) @ base_function(u_rho)   #z_num × (m2+k2-1)


        llambd2 = betalambd @ w.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * ((base_function_dao(w_beta)@eta)**2)) @ w @ betalambd.T \
                + betalambd @ w.T @ np.diag(ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao2(w_beta)@eta)) @ w @ betalambd.T \
                + ((betalambd2@w.T) @ (ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao(w_beta)@eta))).reshape(w_num-1,w_num-1) \
                + betalambd @ w.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * ((base_function_dao(w_beta)@eta)**2)) @ w @ betalambd.T \
                + betalambd @ w.T @ np.diag((1-ind) * ((1/v)*cosh2*d) * (base_function_dao2(w_beta)@eta)) @ w @ betalambd.T \
                + ((betalambd2@w.T) @ ((1-ind) * ((1/v)*cosh2*d) * (base_function_dao(w_beta)@eta))).reshape(w_num-1,w_num-1)   #(w_num-1)×(w_num-1)
        llambds = betalambd @ w.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(w_beta)@eta)) @ x \
                + betalambd @ w.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(w_beta)@eta)) @ x   # (w_num-1) x x_num
        llambdtau = betalambd @ w.T@ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(u_rho)@zeta) * (base_function_dao(w_beta)@eta)) @u @ rhotau.T \
                  + betalambd @ w.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(u_rho)@zeta) * (base_function_dao(w_beta)@eta)) @ u @rhotau.T   # (w_num-1) x (u_num-1)
        llambdeta = betalambd @ w.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * (base_function_dao(w_beta) @ eta)) @ base_function(w_beta) \
                  + betalambd @ w.T @ np.diag(ind * (-0.5*tanh2+(1/(v**2))*sinh1)) @ base_function_dao(w_beta)  \
                  + betalambd @ w.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * (base_function_dao(w_beta) @ eta)) @ base_function(w_beta) \
                  + betalambd @ w.T @ np.diag((1-ind) * ((1/v)*cosh2*d)) @ base_function_dao(w_beta)    #(w_num-1) × (m1+k1-1)
        llambdzeta = betalambd @ w.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(w_beta)@eta)) @ base_function(u_rho) \
                   + betalambd @ w.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(w_beta)@eta)) @ base_function(u_rho)   # (w_num-1) x (m2+k2-1)


        ls2 = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @x \
            + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @x   # x_num x x_num
        lstau = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * (base_function_dao(u_rho)@zeta)) @ u @ rhotau.T \
              + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * (base_function_dao(u_rho)@zeta)) @ u @ rhotau.T   # x_num x (u_num-1)
        lseta = x.T @ np.diag(ind * ((-2/(v**2))*sinh1)) @ base_function(w_beta) \
              + x.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2)) @ base_function(w_beta)   # x_num x (m1+k1-1)
        lszeta = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @base_function(u_rho) \
            + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @base_function(u_rho)   # x_num x (m2+k2-1)


        ltau2 = rhotau @ u.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * ((base_function_dao(u_rho)@zeta)**2)) @u@rhotau.T \
              + rhotau @ u.T @ np.diag(ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao2(u_rho)@zeta)) @u@rhotau.T \
              + ((rhotau2@u.T) @ (ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao(u_rho)@zeta))).reshape(u_num-1,u_num-1) \
              + rhotau @ u.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * ((base_function_dao(u_rho)@zeta)**2)) @u@rhotau.T \
              + rhotau @ u.T @ np.diag((1-ind) * (d*sinh2*(2/v)) * (base_function_dao2(u_rho)@zeta)) @u@rhotau.T \
              + ((rhotau2@u.T) @ ((1-ind)*(d*sinh2*(2/v)) * (base_function_dao(u_rho)@zeta))).reshape(u_num-1,u_num-1)   # (u_num-1) x (u_num-1)
        ltaueta = rhotau @ u.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(u_rho)@zeta)) @ base_function(w_beta) \
                + rhotau @ u.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(u_rho)@zeta)) @ base_function(w_beta)   # (u_num-1) x (m1+k1-1)
        ltauzeta = rhotau @ u.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * (base_function_dao(u_rho)@zeta)) @ base_function(u_rho) \
                 + rhotau @ u.T @ np.diag(ind * ((4/(v**2))*(sinh2**2)-1)) @ base_function_dao(u_rho) \
                 + rhotau @ u.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * (base_function_dao(u_rho)@zeta)) @ base_function(u_rho) \
                 + rhotau @ u.T @ np.diag((1-ind) * (d*sinh2*(2/v))) @ base_function_dao(u_rho)   # (u_num-1) x (m2+k2-1)


        leta2 = base_function(w_beta).T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ base_function(w_beta) \
              + base_function(w_beta).T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ base_function(w_beta)   # (m1+k1-1) x (m1+k1-1)
        letazeta = base_function(w_beta).T @ np.diag(ind * ((-2/(v**2))*sinh1) *v) @ base_function(u_rho) \
                 + base_function(w_beta).T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) *v) @ base_function(u_rho)   # (m1+k1-1) x (m2+k2-1)
        lzeta2 = base_function(u_rho).T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @base_function(u_rho) \
               + base_function(u_rho).T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @base_function(u_rho)   # (m2+k2-1) x (m2+k2-1)

        ltheta = np.hstack((lgamma.T, llambd.T, ls.T, ltau.T, leta.T, lzeta.T)).T
        # print('ltheta=', ltheta, max(abs(ltheta)))

        ltheta2 = np.block([[lgamma2,          lgammalambd,   lgammas,     lgammatau,     lgammaeta,     lgammazeta],
                            [lgammalambd.T,    llambd2,       llambds,     llambdtau,     llambdeta,     llambdzeta],
                            [lgammas.T,        llambds.T,     ls2,         lstau,         lseta,         lszeta],
                            [lgammatau.T,      llambdtau.T,   lstau.T,     ltau2,         ltaueta,       ltauzeta],
                            [lgammaeta.T,      llambdeta.T,   lseta.T,     ltaueta.T,     leta2,         letazeta],
                            [lgammazeta.T,     llambdzeta.T,  lszeta.T,    ltauzeta.T,    letazeta.T,    lzeta2]])


        SE = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
        # print('SE=', SE)
        likelihood_value = (ind@(log(1/sqrt(2*pi))+log(1/v)+log(np.cosh((y-(z@gamma+base_function(w_beta)@eta))/2))\
                       -(2/(v**2))*np.sinh((y-(z@gamma+base_function(w_beta)@eta))/2)**2)\
                       +(1-ind)@(log(1-standardN.cdf((2/v)*np.sinh((y-(z@gamma+base_function(w_beta)@eta))/2)))) )
        # print('likelihood_value =', likelihood_value)
        AIC = -2*likelihood_value+2*len(ltheta)
        # print('AIC =', AIC)

# -------------------------------------------------------------------------

        Theta1 = np.hstack((gamma.T, lambd.T, s.T, tau.T, eta.T, zeta.T)).T
        # print('Overall Theta1 =', Theta1)
        # Thre = max(abs(Theta1 - Theta))/max(abs(Theta))
        # if cond1 < 1e-4 and cond2 < 3e-3:

        Thre = max(abs(Theta1 - Theta))
        print('Overall step size:', Thre)
        if Thre < 1e-4:
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

            iter += 1
            break


print('gamma_m = ', gamma_m, gamma_m.shape)
print('beta_m = ', beta_m, beta_m.shape)
print('s_m = ', s_m, s_m.shape)
print('rho_m = ', rho_m, rho_m.shape)
print('eta_m = ', eta_m, eta_m.shape)
print('zeta_m = ', zeta_m, zeta_m.shape)

gamma_95low = np.percentile(gamma_m, 2.5, axis=0)
gamma_95upper = np.percentile(gamma_m, 97.5, axis=0)
beta_95low = np.percentile(beta_m, 2.5, axis=0)
beta_95upper = np.percentile(beta_m, 97.5, axis=0)
s_95low = np.percentile(s_m, 2.5, axis=0)
s_95upper = np.percentile(s_m, 97.5, axis=0)
rho_95low = np.percentile(rho_m, 2.5, axis=0)
rho_95upper = np.percentile(rho_m, 97.5, axis=0)



# # # -------------------------------- compute standard errors -----------------------------------
# # second derivatives
# lgamma2 = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1))) @ z \
#         + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d))) @ z   # z_num x z_num
# lgammabeta = z.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * (base_function_dao(w_beta)@eta)) @ w  \
#             + z.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * (base_function_dao(w_beta) @ eta)) @ w    # z_num x (w_num-1)
# lgammas = z.T @ np.diag(ind * ((-2/(v**2))*sinh1)) @ x \
#         + z.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2)) @ x   #z_num × x_num
# lgammarho = z.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(u_rho)@zeta)) @u  \
#           + z.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(u_rho)@zeta)) @u    #z_num × (u_num-1)
#
# lbeta2 = w.T @ np.diag(ind * ((0.25/(cosh2**2))-((1/(v**2))*cosh1)) * ((base_function_dao(w_beta)@eta)**2)) @ w \
#         + w.T @ np.diag(ind * (-0.5*tanh2+(1/(v**2))*sinh1) * (base_function_dao2(w_beta)@eta)) @ w  \
#         + w.T @ np.diag((1-ind) * (((-1/(v**2))*d_dao*(cosh2**2))-((-0.5/v)*sinh2*d)) * ((base_function_dao(w_beta)@eta)**2)) @ w  \
#         + w.T @ np.diag((1-ind) * ((1/v)*cosh2*d) * (base_function_dao2(w_beta)@eta)) @ w    #(w_num-1)×(w_num-1)
# lbetas = w.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(w_beta)@eta)) @ x \
#         + w.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(w_beta)@eta)) @ x   # (w_num-1) x x_num
# lbetarho = w.T @ np.diag(ind * ((-2/(v**2))*sinh1) * (base_function_dao(u_rho)@zeta) * (base_function_dao(w_beta)@eta)) @ u  \
#           + w.T @ np.diag((1-ind) * (-d_dao*(1/(v**2))*sinh1-d*(1/v)*cosh2) * (base_function_dao(u_rho)@zeta) * (base_function_dao(w_beta)@eta)) @ u    # (w_num-1) x (u_num-1)
#
# ls2 = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2))) @x \
#     + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2)) @x   # x_num x x_num
# lsrho = x.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * (base_function_dao(u_rho)@zeta)) @ u  \
#       + x.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * (base_function_dao(u_rho)@zeta)) @ u   # x_num x (u_num-1)
#
# lrho2 = u.T @ np.diag(ind * ((-8/(v**2))*(sinh2**2)) * ((base_function_dao(u_rho)@zeta)**2)) @u \
#       + u.T @ np.diag(ind * ((4/(v**2))*(sinh2**2)-1) * (base_function_dao2(u_rho)@zeta)) @u \
#       + u.T @ np.diag((1-ind) * (-d_dao*(4/(v**2))*(sinh2**2)-d*(2/v)*sinh2) * ((base_function_dao(u_rho)@zeta)**2)) @u \
#       + u.T @ np.diag((1-ind) * (d*sinh2*(2/v)) * (base_function_dao2(u_rho)@zeta)) @u   # (u_num-1) x (u_num-1)
#
# ltheta2 = np.block([[lgamma2,          lgammabeta,   lgammas,     lgammarho],
#                     [lgammabeta.T,     lbeta2,       lbetas,      lbetarho],
#                     [lgammas.T,        lbetas.T,     ls2,         lsrho],
#                     [lgammarho.T,      lbetarho.T,   lsrho.T,     lrho2]])
#
# SE_repara = np.sqrt(np.diag(-(np.linalg.inv(ltheta2))))
# print('SE before reparameterization =', SE_repara)
#
#
# gamma_SE = SE[0:z_num]
# s_SE = SE[z_num+w_num-1:z_num+w_num-1+x_num]
# eta_SE = SE[-(nodenum1+degree1-1+nodenum2+degree2-1):-(nodenum2+degree2-1)]
# zeta_SE = SE[-(nodenum2+degree2-1):]
# beta_SE = SE_repara[z_num:z_num+w_num]
# rho_SE = SE_repara[-u_num:]


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
for i in range(m):
    u1 = np.sort(w @ beta_m[i].T).flatten()
    w_beta = np.sort(w @ beta_m[i].T).flatten()  
    g_pre = base_function(w_beta) @ eta_m[i].T
    u2 = np.sort(w_true @ beta_true).flatten()
    w_beta = np.sort(w_true @ beta_true).flatten()
    g_true = base_function(w_beta) @ eta_true

    ax.set_title(f"model2_mean, nodenum1={nodenum1}, degree1={degree1}, n={n}, censor_rate={censor_rate:.2f}")
    plt.xlabel('$\mathrm{W^T\hat{\\beta}}$')
    plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')

    ax.plot(u1[:-1], g_pre[:-1], '--b')  # plot of the estimated mean
    ax.plot(u2[:-1], g_true[:-1], '-r')
plt.show()


fig1_95, ax = plt.subplots()
for i in range(m):
    u1 = np.sort(w @ beta_m[i].T).flatten()
    w_beta = np.sort(w @ beta_m[i].T).flatten()
    g_func_m[i] = base_function(w_beta) @ eta_m[i].T

u2 = np.sort(w_true @ beta_true).flatten()
w_beta = np.sort(w_true @ beta_true).flatten()
g_true = base_function(w_beta) @ eta_true
g_pre_low = np.percentile(g_func_m, 2.5, axis=0)
g_pre_upper = np.percentile(g_func_m, 97.5, axis=0)
ax.set_title(f"model2_mean, nodenum1={nodenum1}, degree1={degree1}, n={n}, censor_rate={censor_rate:.2f}")
plt.xlabel('$\mathrm{W^T\hat{\\beta}}$')
plt.ylabel('$\mathrm{\hat{g}(W^T\hat{\\beta})}$')

ax.plot(u2[:-1], g_true[:-1], '-r')
ax.plot(u2[:-1], g_pre_low[:-1], '--r')
ax.plot(u2[:-1], g_pre_upper[:-1], '--r')
lines, = plt.plot([], [], '-r', label='Estimated curve')
lines2, = plt.plot([], [], '--r', label='95% C.I')
# plt.legend(loc='upper left', frameon=False)
plt.legend()
plt.show()


fig2, ax = plt.subplots()
for j in range(m):
    u3 = np.sort(u @ rho_m[j].T).flatten()
    u_rho = np.sort(u @ rho_m[j].T).flatten()
    alpha_pre = base_function(u_rho) @ zeta_m[j].T
    u4 = np.sort(u_true @ rho_true).flatten()
    u_rho = np.sort(u_true @ rho_true).flatten()
    alpha_true = base_function(u_rho) @ zeta_true

    ax.set_title(f"model2_shape, nodenum2={nodenum2}, degree2={degree2}, n={n}, censor_rate={censor_rate:.2f}")
    plt.xlabel('$\mathrm{U^T\hat{\\rho}}$')
    plt.ylabel('$\mathrm{\hat{\\psi}(U^T\hat{\\rho})}$')

    ax.plot(u3[:-1], alpha_pre[:-1], '--b')
    ax.plot(u4[:-1], alpha_true[:-1], '-r')
plt.show()

fig2_95, ax = plt.subplots()
for j in range(m):
    u3 = np.sort(u @ rho_m[j].T).flatten()
    u_rho = np.sort(u @ rho_m[j].T).flatten()
    alpha_func_m[j] = base_function(u_rho) @ zeta_m[j].T
u4 = np.sort(u_true @ rho_true).flatten()
u_rho = np.sort(u_true @ rho_true).flatten()
alpha_true = base_function(u_rho) @ zeta_true
alpha_pre_low = np.percentile(alpha_func_m, 2.5, axis=0)
alpha_pre_upper = np.percentile(alpha_func_m, 97.5, axis=0)

ax.set_title(f"model2_shape, nodenum2={nodenum2}, degree2={degree2}, n={n}, censor_rate={censor_rate:.2f}")
plt.xlabel('$\mathrm{U^T\hat{\\rho}}$')
plt.ylabel('$\mathrm{\hat{\\psi}(U^T\hat{\\rho})}$')

ax.plot(u4[:-1], alpha_true[:-1], '-r')
ax.plot(u4[:-1], alpha_pre_low[:-1], '--r')
ax.plot(u4[:-1], alpha_pre_upper[:-1], '--r')
lines, = plt.plot([], [], '-r', label='Estimated curve')
lines2, = plt.plot([], [], '--r', label='95% C.I')
# plt.legend(loc='upper left', frameon=False)
plt.legend()
plt.show()



# ------------------------------------- Save data and figures -----------------------------------------
folder_name = os.path.join(os.path.dirname(__file__), 'results', f'model2,m={m},n={n},censor_rate={censor_rate:.4f}')
os.makedirs(folder_name, exist_ok=True)

# save figures
fig_path = os.path.join(folder_name, f'application_model2_mean_single_index,nodenum1={nodenum1},m={m},n={n},censor_rate={censor_rate:.3f}.png')
fig1.savefig(fig_path, dpi=600)
fig_path = os.path.join(folder_name, f'application_model2_mean_CI_band,nodenum1={nodenum1},m={m},n={n},censor_rate={censor_rate:.3f}.png')
fig1_95.savefig(fig_path, dpi=600)
fig_path = os.path.join(folder_name, f'application_model2_shape_single_index,nodenum2={nodenum2},m={m},n={n},censor_rate={censor_rate:.3f}.png')
fig2.savefig(fig_path, dpi=600)
fig_path = os.path.join(folder_name, f'application_model2_shape_CI_band,nodenum2={nodenum2},m={m},n={n},censor_rate={censor_rate:.3f}.png')
fig2_95.savefig(fig_path, dpi=600)


# save data
data_path = os.path.join(folder_name, f'application_model2_results,n={n},censor_rate={censor_rate:.4f}.xlsx')
df1 = pd.DataFrame(g_func_m.reshape(n,m))
df2 = pd.DataFrame(alpha_func_m.reshape(n,m))
data = np.hstack([g_true.reshape(n,1),g_pre_low.reshape(n,1),g_pre_upper.reshape(n,1)])
df3 = pd.DataFrame(data, columns=['g_pre','g_pre_low','g_pre_upper'])
data = np.hstack([alpha_true.reshape(n,1),alpha_pre_low.reshape(n,1),alpha_pre_upper.reshape(n,1)])
df4 = pd.DataFrame(data, columns=['alpha_pre','alpha_pre_low','alpha_pre_upper'])

df5_data = np.hstack([gamma_m, beta_m, s_m, rho_m, eta_m, zeta_m])
df5 = pd.DataFrame(df5_data, columns=['γ1','γ2','γ3','γ4','γ5','γ6','γ7','β1','β2',
                                      's1','s2','s3','s4','s5','s6','s7','ρ1','ρ2',
                                      'η1','η2','η3','η4','η5','η6',
                                      'ζ1','ζ2','ζ3','ζ4','ζ5'])

max_len = max(len(gamma),len(beta),len(s),len(rho))
gamma_len = max_len-len(gamma)
beta_len = max_len-len(beta)
s_len = max_len-len(s)
rho_len = max_len-len(rho)
df6 = pd.DataFrame({'γ_95low':list(gamma_95low)+[None]*gamma_len, 'γ_95upper':list(gamma_95upper)+[None]*gamma_len,'γ_true':list(gamma_true)+[None]*gamma_len,
                    'β_95low':list(beta_95low)+[None]*beta_len,   'β_95upper':list(beta_95upper)+[None]*beta_len,  'β_true':list(beta_true)+[None]*beta_len,
                    's_95low':list(s_95low)+[None]*s_len,         's_95upper':list(s_95upper)+[None]*s_len,        's_true':list(s_true)+[None]*s_len,
                    'ρ_95low':list(rho_95low)+[None]*rho_len,     'ρ_95upper':list(rho_95upper)+[None]*rho_len,    'ρ_true':list(rho_true)+[None]*rho_len
                    })

with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
    df1.to_excel(writer, index=False, sheet_name='g_single_index_each_iter')
    df2.to_excel(writer, index=False, sheet_name='psi_single_index_each_iter')
    df3.to_excel(writer, index=False, sheet_name='g_95pct_CI_band')
    df4.to_excel(writer, index=False, sheet_name='psi_95pct_CI_band')
    df5.to_excel(writer, index=False, sheet_name='estimates')
    df6.to_excel(writer, index=False, sheet_name='95pct_param_CI')


# ---------------------------------------- Compute the running time -----------------------------------------
end_time = time.time()  # record the end time of the program
run_time = end_time - start_time  # compute the running time
print("Running time:", int(run_time // 60), "min", int(run_time % 60), "s")
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 09:06:47 2023

@author: Zhao
"""
import sys
import numpy as np
#import pandas as pd

import dcor as dc
from scipy.optimize import linear_sum_assignment
import time
#import Ball

from statsmodels.sandbox.distributions import multivariate
#import matplotlib.pyplot as plt
#from sklearn import preprocessing


#import seaborn as sns
from scipy.stats import rankdata
from scipy.stats import qmc
#from scipy.stats import kendalltau
#from scipy.stats import pearsonr
import os

# sample size
n = 100
# predictor size
p = 1000
# if multi-dimension, define dim
dim = 10
# set simulation repeat time
rep = 10

# for 1-dimension, to get the rank
def onedranknew(x):
    order = x.argsort()
    r = np.empty_like(order)
    r[order] = np.arange(len(x))
    return(r)

# for multi-dimension, to get the rank
def multi2(x,h):
    b2=np.array([np.sum((x[i]-h)**2,axis=1) for i in range(x.shape[0])])
    c2=linear_sum_assignment(b2)
    return c2[1]

# 1-dimension rank, use equally spaced 
equals = np.linspace(0,1, n)


# below is an illustration simulation with 1-dimension, 
# define mean vector and covariance structure
me=np.zeros(p)
sigma=np.zeros((p,p))
for i,j in np.ndindex(sigma.shape):
    sigma[i,j]=0.5**abs(i-j)

# suppose we have 4 truth
myrank=np.zeros((rep,4))
mystat = np.zeros((rep,p))

start_time = time.time()
for j in range(rep):
    x=multivariate.multivariate_t_rvs(me, sigma,1,n)
    beta = np.random.uniform(2,5,4)
    truex = np.transpose(np.array([x[:,0],x[:,5],x[:,11]**2,x[:,21]]))
    y=truex @ (beta) + np.random.standard_t(df = 1,size = n)
    yassign=equals[onedranknew(y)]
    mystat=np.apply_along_axis(lambda x: dc.distance_correlation_sqr(equals[onedranknew(x)], yassign) , 0, x)
    myrank[j]=p+1-rankdata(mystat)[[0,5,11,21]]
print("--- %s seconds ---" % (time.time() - start_time))


# multi-dimension rank, use qmc sequence, here try Sobol, you can use other like Halton
# s10 is 10-dimensional Sobol, s3 is 3-dimensional Sobol
sampler10 = qmc.Sobol(d=dim, scramble=True, seed = 1234)
sample10 = sampler10.random_base2(m = 9)
s10 = sample10[range(n)]
sampler = qmc.Sobol(d=3, scramble=True, seed = 123)
sample = sampler.random_base2(m = 9)
s3 = sample[range(n)]

#for multi-dimension, just construct multivariate x and y, then use multi2 function to get the rank
L=np.linalg.cholesky(sigma)

# suppose again we have 4 truth, myrank will be the rank of truth using our method
myrank=np.zeros((rep,4))
mystat = np.zeros((rep,p))

start_time = time.time()
for j in range(rep):
    u=(np.random.pareto(10, size = (n,p))+1)*15
    v=(np.random.pareto(10,size=(n,p))+1)*5
    w=(np.random.pareto(12,size=(n,p))+1)*30
    uu = u @ L
    vv = v @ L
    ww = w @ L
    x = np.array([uu,vv,ww])
    idx1 = np.random.choice(range(3),size = 4)
    idx2 = np.random.choice(range(3),size = 4)
    idx3 = np.random.choice(range(3),size = 4)
    idx4 = np.random.choice(range(3),size = 4)
    y3 =(np.random.uniform(1,2,1)* x[idx1,:,1]+np.random.uniform(1,2,1)* x[idx2,:,2]+np.random.uniform(1,2,1)*x[idx3,:,100]+np.random.uniform(1,2,1)*x[idx4,:,101]**2+(np.random.pareto(10,size = (4,n))+1)*15).T
    y4 = (np.random.pareto(10,(n,6))+1)*15
    y = np.column_stack([y3,y4])
    yassign=s10[multi2(y,s10)]
    mystat=np.array( [dc.distance_correlation_sqr(s3[multi2(x[:,:,i].T,s3)], yassign) for i in range(p)])
    myrank[j]=p+1-rankdata(mystat)[[1,2,100,101]]
print("--- %s seconds ---" % (time.time() - start_time))
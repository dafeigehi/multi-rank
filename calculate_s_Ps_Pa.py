# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:46:01 2020

@author: dafeige
"""
import os
import numpy as np
import pandas as pd
import statistics as st
import dcor as dc
from scipy.optimize import linear_sum_assignment
import time
from scipy.stats import rankdata
import scipy.stats
from matplotlib.pyplot import boxplot
from statsmodels.sandbox.distributions import multivariate
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sobol_seq as sb
import seaborn as sns
from statsmodels.tools.sequences import halton
r = 100
p = 5000
n = 200
myrank = np.zeros((4,r))
dcrank = np.zeros((4,r))
rrank = np.zeros((4,r))
sirsrank = np.zeros((4,r))
folder = ""
name = "Mixture-Normal, Beta(1), Weight(1), Para(1), true x(1).csv" 
name = "Mixture-Normal, Beta(1), Weight(1), Para(3), true x(1).csv"
name = "Mixture-Normal, Beta(2), Weight(1), Para(1), true x(1).csv"
name = "Mixture-Normal, Beta(2), Weight(1), Para(3), true x(1).csv"
name = "Mixture-Normal, Beta(1), Weight(2), Para(3), true x(2).csv"
name = "Mixture-Normal, Beta(2), Weight(2), Para(1), true x(2).csv"
name = "Mixture-Normal, Beta(2), Weight(2), Para(3), true x(2).csv"
name = "Mixture-Normal, Beta(1), Weight(2), Para(1), true x(1).csv"
mystat = np.loadtxt(folder+'mystat'+name, delimiter = ',')
dcstat = np.loadtxt(folder+'dcstat'+name, delimiter = ',')
corstat = np.loadtxt(folder+'corstat'+name, delimiter = ',')
sirstat = np.loadtxt(folder+'sirstat'+name, delimiter = ',')

stat = pd.read_csv("F:/study/math/paper/simu/t10000/17.6.out",header = None,engine = 'python', delimiter = '[ ]+').to_numpy()
mystat0 = stat[range(2000),:]
myrank = []
dcrank = []
for file in os.listdir("F:/study/math/paper/simu/t10000/"):
    stat = pd.read_csv(file,header = None,engine = 'python', delimiter = '[ ]+').to_numpy()
    mystat = stat[range(2000),:]
    dcstat = stat[range(2000,4000),:]
    myrank.append(p*2000+1-rankdata(mystat.flatten())[[1,3,p+1,p+3,p*2+1,p*2+3]])
    dcrank.append(p*2000+1-rankdata(dcstat.flatten())[[1,3,p+1,p+3,p*2+1,p*2+3]])
for i, file in enumerate(os.listdir("F:/study/math/paper/simu/t10000/")):
    stat = pd.read_csv(file,header = None,engine = 'python', delimiter = '[ ]+').to_numpy()
    mystat = stat[range(2000),:]
    dcstat = stat[range(2000,4000),:]
    myrank[:,i] = p*2000+1-rankdata(mystat.flatten())[[1,3,p+1,p+3,p*2+1,p*2+3]]
    dcrank[:,i] = p*2000+1-rankdata(dcstat.flatten())[[1,3,p+1,p+3,p*2+1,p*2+3]]

myrank1 = myrank[:,range(55)]
dcrank1 = dcrank[:,range(55)]
dcs = np.amax(dcrank1,0)
mys = np.amax(myrank1,0)
dcs_quantile = np.quantile(dcs, [0.25,0.5,0.75,0.95])
mys_quantile = np.quantile(mys, [0.25,0.5,0.75,0.95])
any_dc_d1 = (dcrank1<=d1).sum(1)/55
any_my_d1 = (myrank1<=d1).sum(1)/55
any_dc_d2 = (dcrank1<=2*d1).sum(1)/55
any_my_d2 = (myrank1<=2*d1).sum(1)/55
any_dc_d3 = (dcrank1<=d1*3).sum(1)/55
any_my_d3 = (myrank1<=d1*3).sum(1)/55
strict_dc_d1 = (dcs<=d1).sum()/55
strict_my_d1 = (mys<=d1).sum()/55
strict_dc_d2 = (dcs<=2*d1).sum()/55
strict_my_d2 = (mys<=2*d1).sum()/55
strict_dc_d3 = (dcs<=3*d1).sum()/55
strict_my_d3 = (mys<=3*d1).sum()/55


for j in range(r):
    myrank[:,j] = p+1-rankdata(mystat[j])[[0,1,11,21]]
    dcrank[:,j] = p+1-rankdata(dcstat[j])[[0,1,11,21]]
    rrank[:,j] = p+1-rankdata(corstat[j])[[0,1,11,21]]
    sirsrank[:,j] = p+1-rankdata(sirstat[j])[[0,1,11,21]]

boxplot(np.transpose(np.concatenate((myrank,dcrank,rrank,sirsrank))))
dcs = np.amax(dcrank,0)
mys = np.amax(myrank,0)
rs = np.amax(rrank, 0)
sirss = np.amax(sirsrank, 0)

rs_quantile = np.quantile(rs, [0.25,0.5,0.75,0.95])
sirss_quantile = np.quantile(sirss, [0.25,0.5,0.75,0.95])
dcs_quantile = np.quantile(dcs, [0.25,0.5,0.75,0.95])
mys_quantile = np.quantile(mys, [0.25,0.5,0.75,0.95])

corstat = np.loadtxt("corstatbeta(2), true x = 4, location = 1, scale = 2.csv",delimiter = ",")
sirstat = np.loadtxt("sirstatbeta(2), true x = 4, location = 1, scale = 2.csv",delimiter = ",")
dcstat = np.loadtxt("dcstatbeta(2), true x = 4, location = 1, scale = 2.csv",delimiter = ",")
mystat = np.loadtxt("mystatbeta(2), true x = 4, location = 1, scale = 2.csv",delimiter = ",")

corstat = np.loadtxt("totalcorstatxx(3), truex(1).csv",delimiter = ",")
sirstat = np.loadtxt("totalsirstatxx(3), truex(1).csv",delimiter = ",")
dcstat = np.loadtxt("totaldcstatxx(3), truex(1).csv",delimiter = ",")
mystat = np.loadtxt("totalmystatxx(3), truex(1).csv",delimiter = ",")

corstat = np.loadtxt("corstatMixture-Normal, Beta(2), Weight(2), Para(3), true x(2).csv",delimiter = ",")
sirstat = np.loadtxt("sirstatMixture-Normal, Beta(2), Weight(2), Para(3), true x(2).csv",delimiter = ",")
dcstat = np.loadtxt("dcstatMixture-Normal, Beta(2), Weight(2), Para(3), true x(2).csv",delimiter = ",")
mystat = np.loadtxt("mystatMixture-Normal, Beta(2), Weight(2), Para(3), true x(2).csv",delimiter = ",")
for j in range(r):
    myrank[:,j] = p+2-rankdata(mystat[j])[[0]]
    dcrank[:,j] = p+2-rankdata(dcstat[j])[[0]]
    rrank[:,j] = p+2-rankdata(corstat[j])[[0]]
    sirsrank[:,j] = p+2-rankdata(sirstat[j])[[0]]
d1 = int(n/np.log(n))

# Calculates Power
# Any Power
d1 = 37

any_r_d1 = (rrank<=d1).sum(1)
any_sirs_d1 = (sirsrank<=d1).sum(1)
any_dc_d1 = (dcrank<=d1).sum(1)
any_my_d1 = (myrank<=d1).sum(1)


any_r_d2 = (rrank<=2*d1).sum(1)
any_sirs_d2 = (sirsrank<=2*d1).sum(1)
any_dc_d2 = (dcrank<=2*d1).sum(1)
any_my_d2 = (myrank<=2*d1).sum(1)


any_r_d3 = (rrank<=d1*3).sum(1)
any_sirs_d3 = (sirsrank<=d1*3).sum(1)
any_dc_d3 = (dcrank<=d1*3).sum(1)
any_my_d3 = (myrank<=d1*3).sum(1)

# Strict Power

strict_r_d1 = (rs<=d1).sum()
strict_sirs_d1 = (sirss<=d1).sum()
strict_dc_d1 = (dcs<=d1).sum()
strict_my_d1 = (mys<=d1).sum()


strict_r_d2 = (rs<=2*d1).sum()
strict_sirs_d2 = (sirss<=2*d1).sum()
strict_dc_d2 = (dcs<=2*d1).sum()
strict_my_d2 = (mys<=2*d1).sum()


strict_r_d3 = (rs<=3*d1).sum()
strict_sirs_d3 = (sirss<=3*d1).sum()
strict_dc_d3 = (dcs<=3*d1).sum()
strict_my_d3 = (mys<=3*d1).sum()
# revise
# t_1_normal
os.chdir('F:\study\math\paper\simu')
myrank = np.loadtxt('1st_revise_t1_1_myrank.txt', delimiter = ',')
corrank = np.loadtxt('1st_revise_t1_1_corrank.txt', delimiter = ',')
sirsrank = np.loadtxt('1st_revise_t1_1_sirsrank.txt', delimiter = ',')
dcrank = np.loadtxt('1st_revise_t1_1_dcrank.txt', delimiter = ',')
dcrorank = np.loadtxt('1st_revise_t1_1_dcrorank.txt', delimiter = ',')
scrank = np.loadtxt('1st_revise_t1_1_scrank.txt', delimiter = ',')
pcrank = np.loadtxt('1st_revise_t1_1_pcrank.txt', delimiter = ',')
rrcsrank = np.loadtxt('1st_revise_t1_1_rrcsrank.txt', delimiter = ',')
bcorrank = np.loadtxt('1st_revise_t1_1_bcorrank.txt', delimiter = ',')

boxplot(np.concatenate((myrank,corrank,sirsrank,dcrank,dcrorank,scrank,pcrank,rrcsrank,bcorrank)))
boxplot(myrank)
myrank.shape
boxplot(np.concatenate((myrank, corrank, sirsrank,dcrank,dcrorank,scrank,pcrank,rrcsrank,bcorrank),axis = 1))

mys = np.amax(myrank, axis = 1, keepdims=True)
cors = np.amax(corrank, axis = 1, keepdims=True)
sirss = np.amax(sirsrank, axis = 1, keepdims=True)
dcs = np.amax(dcrank, axis = 1, keepdims=True)
dcros = np.amax(dcrorank, axis = 1, keepdims=True)
scs = np.amax(scrank, axis = 1, keepdims=True)
pcs = np.amax(pcrank, axis = 1, keepdims=True)
rrcss = np.amax(rrcsrank, axis = 1, keepdims=True)
bcors = np.amax(bcorrank, axis = 1, keepdims=True)

cors.mean(), cors.std()
sirss.mean(), sirss.std()
rrcss.mean(), rrcss.std()
scs.mean(), scs.std()
pcs.mean(), pcs.std()
bcors.mean(), bcors.std()
dcs.mean(), dcs.std()
dcros.mean(), dcros.std()
mys.mean(), mys.std()

boxplot(np.concatenate((mys, cors, sirss,dcs,dcros,scs,pcs,rrcss,bcors),axis = 1))
tot = pd.DataFrame(np.concatenate((cors, sirss,rrcss,scs,pcs,bcors,dcs,dcros,mys),axis = 1),columns = ['SIS', 'SIRS', 'RRCS', 'SC-SIS', 'PC-Screen', 'BCor-SIS','DC-SIS', 'DC-RoSIS', 'MrDc-SIS'])
plt1 = sns.boxplot(data = tot)
plt1.set_ylabel('Min Model Size to Include All Truth', size = 20)
plt1.tick_params(labelsize = 22)

#P_s
d1 = 37
np.sum(corrank<d1, axis = 0)/200
np.sum(sirsrank<d1, axis = 0)/200
np.sum(rrcsrank<d1, axis = 0)/200
np.sum(scrank<d1, axis = 0)/200
np.sum(pcrank<d1, axis = 0)/200
np.sum(bcorrank<d1, axis = 0)/200
np.sum(dcrank<d1, axis = 0)/200
np.sum(dcrorank<d1, axis = 0)/200
np.sum(myrank<d1, axis = 0)/200

np.sum(corrank<d1*2, axis = 0)/200
np.sum(sirsrank<d1*2, axis = 0)/200
np.sum(rrcsrank<d1*2, axis = 0)/200
np.sum(scrank<d1*2, axis = 0)/200
np.sum(pcrank<d1*2, axis = 0)/200
np.sum(bcorrank<d1*2, axis = 0)/200
np.sum(dcrank<d1*2, axis = 0)/200
np.sum(dcrorank<d1*2, axis = 0)/200
np.sum(myrank<d1*2, axis = 0)/200

np.sum(corrank<d1*3, axis = 0)/200
np.sum(sirsrank<d1*3, axis = 0)/200
np.sum(rrcsrank<d1*3, axis = 0)/200
np.sum(scrank<d1*3, axis = 0)/200
np.sum(pcrank<d1*3, axis = 0)/200
np.sum(bcorrank<d1*3, axis = 0)/200
np.sum(dcrank<d1*3, axis = 0)/200
np.sum(dcrorank<d1*3, axis = 0)/200
np.sum(myrank<d1*3, axis = 0)/200
# P_a
np.sum(cors<d1)/200
np.sum(sirss<d1)/200
np.sum(rrcss<d1)/200
np.sum(scs<d1)/200
np.sum(pcs<d1)/200
np.sum(bcors<d1)/200
np.sum(dcs<d1)/200
np.sum(dcros<d1)/200
np.sum(mys<d1)/200

np.sum(cors<d1*2)/200
np.sum(sirss<d1*2)/200
np.sum(rrcss<d1*2)/200
np.sum(scs<d1*2)/200
np.sum(pcs<d1*2)/200
np.sum(bcors<d1*2)/200
np.sum(dcs<d1*2)/200
np.sum(dcros<d1*2)/200
np.sum(mys<d1*2)/200

np.sum(cors<d1*3)/200
np.sum(sirss<d1*3)/200
np.sum(rrcss<d1*3)/200
np.sum(scs<d1*3)/200
np.sum(pcs<d1*3)/200
np.sum(bcors<d1*3)/200
np.sum(dcs<d1*3)/200
np.sum(dcros<d1*3)/200
np.sum(mys<d1*3)/200

# t1_t1
myrank = np.loadtxt('1st_revise_t1_t1_myrank.txt')
corrank = np.loadtxt('1st_revise_t1_t1_corrank.txt')
sirsrank = np.loadtxt('1st_revise_t1_t1_sirsrank.txt')
dcrank = np.loadtxt('1st_revise_t1_t1_dcrank.txt')
dcrorank = np.loadtxt('1st_revise_t1_t1_dcrorank.txt')
scrank = np.loadtxt('1st_revise_t1_t1_scrank.txt')
pcrank = np.loadtxt('1st_revise_t1_t1_pcrank.txt')
rrcsrank = np.loadtxt('1st_revise_t1_t1_rrcsrank.txt')
bcorrank = np.loadtxt('1st_revise_t1_t1_bcorrank.txt')


# multi
myrank = np.loadtxt('1st_revise_multi_t_myrank.txt')

dcrank = np.loadtxt('1st_revise_multi_t_dcrank.txt')

scrank = np.loadtxt('1st_revise_multi_t_scrank.txt')
pcrank = np.loadtxt('1st_revise_multi_t_pcrank.txt')

bcorrank = np.loadtxt('1st_revise_multi_t_bcorrank.txt')

np.sum(scrank<d1*1, axis = 0)/200
np.sum(pcrank<d1*1, axis = 0)/200
np.sum(bcorrank<d1*1, axis = 0)/200
np.sum(dcrank<d1*1, axis = 0)/200
np.sum(myrank<d1*1, axis = 0)/200

np.sum(scs<d1*1)/200
np.sum(pcs<d1*1)/200
np.sum(bcors<d1*1)/200
np.sum(dcs<d1*1)/200
np.sum(mys<d1*1)/200
tot = pd.DataFrame(np.concatenate((scs,pcs,bcors,dcs,mys),axis = 1),columns = ['SC-SIS', 'PC-Screen', 'BCor-SIS','DC-SIS', 'MRDC-SIS'])
plt1 = sns.boxplot(data = tot)
plt1.set_ylabel('Min Model Size to Include All Truth', size = 20)
plt1.tick_params(labelsize = 22)
mys.mean(), mys.std()
dcs.mean(), dcs.std()
scs.mean(), scs.std()
pcs.mean(), pcs.std()
bcors.mean(), bcors.std()

#multi pareto with one dim  binomial
os.chdir('F:\study\math\paper\simu')
myrank = np.loadtxt('1st_revise_multi_pareto_binomial_myrank.txt')
dcrank = np.loadtxt('1st_revise_multi_pareto_binomial_dcrank.txt')
scrank = np.loadtxt('1st_revise_multi_pareto_binomial_scrank.txt')
pcrank = np.loadtxt('1st_revise_multi_pareto_binomial_pcrank.txt')
bcorrank = np.loadtxt('1st_revise_multi_pareto_binomial_bcorrank.txt')

#multi pareto pure
os.chdir('F:\study\math\paper\simu')
myrank = np.loadtxt('1st_revise_multi_pareto_myrank.txt')
dcrank = np.loadtxt('1st_revise_multi_pareto_dcrank.txt')
scrank = np.loadtxt('1st_revise_multi_pareto_scrank.txt')
pcrank = np.loadtxt('1st_revise_multi_pareto_pcrank.txt')
bcorrank = np.loadtxt('1st_revise_multi_pareto_bcorrank.txt')

#multi pareto with both predictor and response contains categorical
os.chdir('F:\study\math\paper\simu')
myrank = np.loadtxt('1st_revise_multi_pareto_two_category_myrank.txt')
dcrank = np.loadtxt('1st_revise_multi_pareto_two_category_dcrank.txt')
scrank = np.loadtxt('1st_revise_multi_pareto_two_category_scrank.txt')
pcrank = np.loadtxt('1st_revise_multi_pareto_two_category_pcrank.txt')
bcorrank = np.loadtxt('1st_revise_multi_pareto_two_category_bcorrank.txt')

# example why we need different measure of dependence
a = np.random.normal(size = 200)
b = np.sqrt(np.abs(0.5 - a**2))
plt.scatter(a,b)
pearsonr(a,b)
kendalltau(a,b)
sc2(a,b)
projection_corr_1d(a,b)
dc.distance_correlation(a,b)
Ball.bcor(a,b)


#3rd revise
#2d y with 1 truth
os.chdir('E:/math/paper/3rd/revise/output')
myrank2d = np.loadtxt('2d_y_myrank.txt', delimiter=',')
dcrank2d = np.loadtxt('2d_y_dcrank.txt', delimiter = ',')
myrank3d = np.loadtxt('3d_y_myrank.txt', delimiter=',')
dcrank3d = np.loadtxt('3d_y_dcrank.txt', delimiter = ',')
myrank4d = np.loadtxt('4d_y_myrank.txt', delimiter=',')
dcrank4d = np.loadtxt('4d_y_dcrank.txt', delimiter = ',')
myrank5d = np.loadtxt('5d_y_myrank.txt', delimiter=',')
dcrank5d = np.loadtxt('5d_y_dcrank.txt', delimiter = ',')
myrank6d = np.loadtxt('6d_y_myrank.txt', delimiter=',')
dcrank6d = np.loadtxt('6d_y_dcrank.txt', delimiter = ',')
myrank7d = np.loadtxt('7d_y_myrank.txt', delimiter=',')
dcrank7d = np.loadtxt('7d_y_dcrank.txt', delimiter = ',')
myrank8d = np.loadtxt('8d_y_myrank.txt', delimiter=',')
dcrank8d = np.loadtxt('8d_y_dcrank.txt', delimiter = ',')
myrank9d = np.loadtxt('9d_y_myrank.txt', delimiter=',')
dcrank9d = np.loadtxt('9d_y_dcrank.txt', delimiter = ',')
myrank10d = np.loadtxt('10d_y_myrank.txt', delimiter=',')
dcrank10d = np.loadtxt('10d_y_dcrank.txt', delimiter = ',')
# myrank15d = np.loadtxt('15d_y_myrank.txt', delimiter=',')
# dcrank15d = np.loadtxt('15d_y_dcrank.txt', delimiter = ',')
myrank20d = np.loadtxt('20d_y_myrank.txt', delimiter=',')
dcrank20d = np.loadtxt('20d_y_dcrank.txt', delimiter = ',')
myrank30d = np.loadtxt('30d_y_myrank.txt', delimiter=',')
dcrank30d = np.loadtxt('30d_y_dcrank.txt', delimiter = ',')
myrank40d = np.loadtxt('40d_y_myrank.txt', delimiter=',')
dcrank40d = np.loadtxt('40d_y_dcrank.txt', delimiter = ',')
myrank50d = np.loadtxt('50d_y_myrank.txt', delimiter=',')
dcrank50d = np.loadtxt('50d_y_dcrank.txt', delimiter = ',')
myrank60d = np.loadtxt('60d_y_myrank.txt', delimiter=',')
dcrank60d = np.loadtxt('60d_y_dcrank.txt', delimiter = ',')
myrank70d = np.loadtxt('70d_y_myrank.txt', delimiter=',')
dcrank70d = np.loadtxt('70d_y_dcrank.txt', delimiter = ',')
myrank80d = np.loadtxt('80d_y_myrank.txt', delimiter=',')
dcrank80d = np.loadtxt('80d_y_dcrank.txt', delimiter = ',')

myrank7d = np.loadtxt('7d_y_myrank.txt', delimiter=',')
dcrank7d = np.loadtxt('7d_y_dcrank.txt', delimiter = ',')
myrank7d = np.loadtxt('7d_y_myrank.txt', delimiter=',')
dcrank7d = np.loadtxt('7d_y_dcrank.txt', delimiter = ',')


d1 = 37

mys = np.amax(myrank80d, axis = 1, keepdims=True)
dcs = np.amax(dcrank80d, axis = 1, keepdims=True)
np.sum(myrank80d<d1, axis = 0)/200
np.sum(mys<d1)/200
np.sum(dcrank80d<d1, axis = 0)/200
np.sum(dcs<d1)/200
np.sum(myrank80d<d1*2, axis = 0)/200
np.sum(mys<d1*2)/200
np.sum(dcrank80d<d1*2, axis = 0)/200
np.sum(dcs<d1*2)/200
np.sum(myrank80d<d1*3, axis = 0)/200
np.sum(mys<d1*3)/200
np.sum(dcrank80d<d1*3, axis = 0)/200
np.sum(dcs<d1*3)/200

boxplot(myrank)
boxplot(dcrank)
boxplot(np.concatenate((myrank, dcrank),axis = 1))
def draw_plot(data, offset,edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset 
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

fig, ax = plt.subplots()
draw_plot(myrank2d, -0.2, "tomato", "white")
draw_plot(myrank3d, -0.1,"tomato", "white")
draw_plot(dcrank2d, +0.1, "skyblue", "white")
draw_plot(dcrank3d, +0.2,"skyblue", "white")

plt.savefig(__file__+'.png', bbox_inches='tight')
plt.show()
plt.close()

# 1d, t1 distribution
myrank = np.loadtxt('t1log_myrank.txt', delimiter=',')
dcrank = np.loadtxt('t1log_dcrank.txt', delimiter = ',')
sisrank = np.loadtxt('t1log_corrank.txt', delimiter=',')
d1 = 37
r = 100
mys = np.amax(myrank, axis = 1, keepdims=True)
dcs = np.amax(dcrank, axis = 1, keepdims=True)
siss = np.amax(sisrank, axis = 1, keepdims=True)
np.sum(myrank<d1, axis = 0)/r
np.sum(mys<d1)/r
np.sum(dcrank<d1, axis = 0)/r
np.sum(dcs<d1)/r
np.sum(sisrank<d1, axis = 0)/r
np.sum(siss<d1)/r
np.sum(myrank<d1*2, axis = 0)/r
np.sum(mys<d1*2)/r
np.sum(dcrank<d1*2, axis = 0)/r
np.sum(dcs<d1*2)/r
np.sum(sisrank<d1*2, axis = 0)/r
np.sum(siss<d1*2)/r
np.sum(myrank<d1*3, axis = 0)/r
np.sum(mys<d1*3)/r
np.sum(dcrank<d1*3, axis = 0)/r
np.sum(dcs<d1*3)/r
np.sum(sisrank<d1*3, axis = 0)/r
np.sum(siss<d1*3)/r

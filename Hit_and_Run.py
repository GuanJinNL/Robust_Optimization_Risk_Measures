#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxpy as cp
import mosek


# In[4]:

#### evaluating the kullback-leibler phi divergence

def kl_calc(q,p):
    N = len(p)
    phi = 0
    for i in range(N):
        if q[i]<= 0:
            phi = phi + 0
        else:
            phi = phi + q[i]*np.log(q[i]/p[i])
    return(phi)

#### Evaluating the modified chi-squared divergence

def mod_chi2_calc(q,p):
    N = len(p)
    phi = 0
    for i in range(N):
        phi = phi + (p[i]-q[i])**2/p[i]
    return(phi)

# Technical functions for the hit-and-run algorithm

def direction(m):
    theta = np.random.normal(0,1,size = m)
    return(theta/np.sum(theta))

def L_maxi(p,q_0,theta,r,phi_calc):
    l_b = 20
    l_a = 0
    l = (l_b+ l_a)/2
    m = len(p)
    while l_b-l_a > 1e-5:
        if np.min(q_0 + l*theta)< -1e-10 or 1-np.sum(q_0+l*theta)<1e-10 or \
        phi_calc(np.concatenate((q_0+l*theta,[1-np.sum(q_0+l*theta)])),p) > r:
            l_b = l
            l = (l_b+l_a)/2
        else:
            l_a = l
            l = (l_b+l_a)/2
    return(l_a)
def L_mini (p,q_0,theta,r,phi_calc):
    l_b = 0
    l_a = -20
    l = (l_b+ l_a)/2
    m = len(p)
    while l_b-l_a > 1e-5:
        if np.min(q_0 + l*theta)< -1e-10 or 1-np.sum(q_0+l*theta)<1e-10 or \
        phi_calc(np.concatenate((q_0+l*theta,[1-np.sum(q_0+l*theta)])),p) > r:
            l_a = l
            l = (l_b+l_a)/2
        else:
            l_b = l
            l = (l_b+l_a)/2
    return(l_b)

#### The hit-and-run algorithm given a phi-divergence function

def hit_and_run(p,phi_calc,r,par,steps):
    N = len(p)
    q_0 = p[0:N-1]
    points = []
    for i in range(steps):
        theta = direction(N-1)
        l_max = L_maxi(p,q_0,theta,r,phi_calc)
        l_min = L_mini (p,q_0,theta,r,phi_calc)
        l_rand = np.random.uniform(l_min,l_max)
        q_new = np.concatenate((q_0+l_rand*theta, [1-np.sum(q_0+l_rand*theta)]))
        points.append(q_new)
        q_0 = q_new[0:N-1]
    return(points)

#### This function calculates the risk evaluation given a probability q and a solution a.

def riskcalc(a,h_eva,R,q,par):
    x = -R.dot(a)
    rank = np.argsort(-x)
    N = len(x)
    q_b = np.zeros(N)
    q_b[0] = h_eva(q[rank[0]],par)
    risk = q_b[0]*x[rank[0]]
    for i in range(1,N):
        q_b[i] = h_eva(np.sum(q[rank[0:i+1]]),par)-h_eva(np.sum(q[rank[0:i]]),par)
        risk = risk + q_b[i]*x[rank[i]]
    return(risk)


#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cvxpy as cp
import mosek
import random
import matplotlib.pyplot as plt
from itertools import chain, combinations


# In[12]:


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def ranktoset (A):
    A = list(A)
    sets = [[A[0]]]
    for i in range(1,len(A)):
        new = A[0:i+1]
        sets.append(new)
    return(sets)

def makesetflex (A, B):    # we assume A is non-empty
    N = len(A)
    B = list(B)
    M = len(B)
    added = []
    for i in range(M):
        new = B[0:i+1]
        for k in range(N):
            if len(A[k])==len(new) and len(np.intersect1d(A[k],new))==len(new):
                break
            if k == N-1:
                A.append(new)
                added.append(new)
    return(A,added)


def countsets(sets):
    m = len(sets)
    count = 0
    for k in range(m):
        count = count + len(sets[k])
    return(count)

def convertlist(sets):
    Output = []
    for temp in sets:
        for elem in temp:
            Output.append(elem)
    return(Output)


def make_psets(N):
    psets = list(powerset(list(range(N))))
    for i in range(1,len(psets)):
        psets[i] = list(psets[i])
    psets = psets[1:(len(psets))]
    return(psets)

def make_RC_Var(N,M,I):
    a = cp.Variable(I)
    v = cp.Variable(M)
    lbda = cp.Variable(M, nonneg = True)
    alpha = cp.Variable(1)
    beta = cp.Variable(1)
    gamma = cp.Variable(1,nonneg = True)
    t = cp.Variable(N)
    z = cp.Variable(M)
    s = cp.Variable(N)
    return(a,v,lbda,alpha,beta,gamma,t,z,s)

def make_cut_Var(N,I,p,h_eva,par):
    a = cp.Variable(I)
    h = np.zeros(N)
    for i in range(N-1):
        h[i] = h_eva(sum(p[i:N]),par)-h_eva(sum(p[i+1:N]),par)
    h[N-1]=h_eva(p[N-1],par)
    return(a,h)


# In[13]:


def robustcheck(x,p,h_func,phi_func,r,par):
    N = len(p)
    rank = np.argsort(x)
    q_b = cp.Variable(N, nonneg = True)
    q = cp.Variable(N, nonneg=True)
    constraints = [cp.sum(q) == 1, cp.sum(q_b)==1]
    constraints = h_func(q_b,q,rank,par,constraints)
    constraints = phi_func(p,q,r,par,constraints)
    obj = cp.Maximize(-q_b.T @ x)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    return(prob.value,q_b.value)

def nominal_risk(x,p,h_eva,par):
    N = len(p)
    rank = np.argsort(x)
    q_b = np.zeros(N)
    q_b[rank[0]] = h_eva(p[rank[0]],par)
    risk = -q_b[rank[0]]*x[rank[0]]
    for i in range(1,N):
        q_b[rank[i]] = h_eva(np.sum(p[rank[0:i+1]]),par)-h_eva(np.sum(p[rank[0:i]]),par)
        risk = risk - q_b[rank[i]]*x[rank[i]]
    return(risk, q_b)


# In[14]:


def cut_rob_pmin(R,p,e_tol,utility,utility_eva,h_func,phi_func,h_eva,r,W0, par=0.5, par_u=1):
    N = len(p)
    I = len(R[0])
    a = cp.Variable(I)
    c = cp.Variable(1)
    constraints = [cp.abs(a)<=W0, cp.sum(a)==1]
    iterations = 0
    constraints.append(-p.T @ utility(R,a,W0,par_u)<= c)
    obj = cp.Minimize(c)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    w = a.value
    while True:
        x = utility_eva(R,w,W0,par_u)
        [rbvalue,h] = robustcheck(x,p,h_func,phi_func,r,par)
        print(rbvalue,c.value,iterations)
        if rbvalue <= c.value+e_tol:
            return(w, prob.value, iterations)
        constraints.append(-h.T @ utility(R,a,W0,par_u)<= c)
        iterations = iterations + 1
        prob = cp.Problem(obj,constraints)
        prob.solve(solver=cp.MOSEK)
        w = a.value
        
def cut_nom_pmin(R,p,e_tol,utility,utility_eva,h_eva,W0,par=1,par_u=1):
    N = len(p)
    I = len(R[0])
    a = cp.Variable(I)
    c = cp.Variable(1)
    constraints = [cp.abs(a)<=W0, cp.sum(a)==1]
    iterations = 0
    constraints.append(-p.T @ utility(R,a,W0,par_u)<= c)
    obj = cp.Minimize(c)
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.ECOS)
    w = a.value
    while True:
        x = utility_eva(R,w,W0,par_u)
        [rbvalue,h] = nominal_risk(x,p,h_eva,par)
        print(rbvalue,c.value,iterations)
        if rbvalue <= c.value+e_tol:
            return(w, prob.value, iterations)
        constraints.append(-h.T @ utility(R,a,W0,par_u)<= c)
        iterations = iterations + 1
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.ECOS)
        w = a.value
        #prob.solve(solver = cp.MOSEK, mosek_params={'MSK_DPAR_BASIS_REL_TOL_S':1e-8})
     
     
def cut_rob_pmax(R,p,e_tol,utility,utility_eva,h_func,phi_func,h_eva,r, r_f, c, W0,par=2, par_u=1):
    N = len(p)
    I = len(R[0])
    a = cp.Variable(I)
    constraints = [cp.abs(a)<=W0]
    iterations = 0
    constraints.append(-p.T @ utility(R,r_f,a,W0,par_u)<= c)
    obj = cp.Maximize(p.T @ utility(R,r_f,a,W0,par_u))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver=cp.MOSEK)
    w = a.value
    while True:
        x = utility_eva(R,r_f,w,W0,par_u)
        [rbvalue,h] = robustcheck(x,p,h_func,phi_func,r,par)
        print(rbvalue,c,iterations)
        if rbvalue <= c+e_tol:
            return(w, prob.value, iterations)
        constraints.append(-h.T @ utility(R,r_f,a,W0,par_u)<= c)
        iterations = iterations + 1
        prob = cp.Problem(obj,constraints)
        prob.solve(solver=cp.MOSEK)
        w = a.value
       
        
def cut_nom_pmax(R,r_f, c, p,e_tol,utility,utility_eva,h_eva,W0,par=2,par_u=1):
    N = len(p)
    I = len(R[0])
    [a,h] = make_cut_Var(N,I,p,h_eva,par)
    constraints = [cp.abs(a)<=W0]
    iterations = 0
    constraints.append(-h.T @ utility(R,r_f,a,W0,par_u)<= c)
    obj = cp.Maximize(p.T @ utility(R,r_f,a,W0,par_u))
    prob = cp.Problem(obj,constraints)
    prob.solve(solver = cp.ECOS)
    w = a.value
    while True:
        x = utility_eva(R,r_f,w,W0,par_u)
        [rbvalue,h] = nominal_risk(x,p,h_eva,par)
        print(rbvalue,c,iterations)
        if rbvalue <= c+e_tol:
            return(w, prob.value, iterations)
        constraints.append(-h.T @ utility(R,r_f,a,W0,par_u)<= c)
        iterations = iterations + 1
        prob = cp.Problem(obj,constraints)
        prob.solve(solver = cp.ECOS)
        w = a.value
        #prob.solve(solver = cp.MOSEK, mosek_params={'MSK_DPAR_BASIS_REL_TOL_S':1e-8})
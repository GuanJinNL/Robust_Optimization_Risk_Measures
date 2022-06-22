#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import cvxpy as cp
import mosek


# In[6]:


####### representation of the epigraph of the perspective of the (-h) conjugates: lambda*((-h)^*)(-v/lambda)<=z

#### quadratic h function h(x) = (1+par)*x - par*(x^2), 0<par<1
def h_quad_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    eta = cp.Variable(M, nonneg= True)
    for j in range(M):
        constraints.append(cp.norm(cp.vstack([eta[j],(z[j]-lbda[j])/2]))<=(z[j]+lbda[j])/2)
        constraints.append(1/(2*np.sqrt(par))*(-v[j]+lbda[j]+par*lbda[j])<= eta[j])
    return(constraints)

#### function h(x) = 1 - (1-x)^par for 0<=x<1, h(x)=1 for x>=1
def h_spw_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    xi_2 = cp.Variable(M, nonneg = True)
    xi_3 = cp.Variable(M, nonneg = True)
    xi_4 = cp.Variable(M, nonneg = True)
    constraints.extend((xi_3 <= xi_2, xi_3 <= v))
    constraints.append(lbda-xi_3+(par**(-1/(par-1))-par**(-par/(par-1)))*xi_4 <= z)
    exponent = np.array([(par-1)/par,1-(par-1)/par])
    for j in range(M):
        constraints.append(xi_2[j]-cp.geo_mean(cp.vstack([xi_4[j],lbda[j]]),exponent)<= 0)
    return(constraints)

#### function h(x) = (1-(1-x)^par)^{1/par} for 0 <=x<1, h(x)=1 for x>=1
def h_dpw_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    xi_2 = cp.Variable(M, nonneg = True)
    xi_3 = cp.Variable(M, nonneg = True)
    constraints.extend((xi_2 >= lbda, -v + xi_3 <=0))
    exp = par/(par-1)
    for j in range(M):
        constraints.append(-xi_3[j]+cp.pnorm(cp.vstack([xi_2[j],xi_3[j]]),exp)<= z[j])
    return(constraints)

#### function h(x) = x^par, 0<par<1
def h_pw_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    constraints.append(v >= 0)
    const = (par**(par/(1-par))-par**(1/(1-par)))**(1-par)
    exp = np.array([1-par, par])
    for j in range(M):
        constraints.append(cp.geo_mean(cp.vstack([z[j],v[j]]),exp)>= lbda[j]*const)
    return(constraints)
#### function h(x) = x^par(1-log(x^par))
def h_log_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    xi_1 = cp.Variable(M, nonneg = True)
    xi_2 = cp.Variable(M, nonneg = True)
    xi_3 = cp.Variable(M, nonneg = True)
    xi_4 = cp.Variable(M, nonneg = True)
    const = (par**(par/(1-par))-par**(1/(1-par)))
    constraints.extend((v >= 0, xi_3 + xi_1 + const* xi_4 <= z))
    exp = np.array([par, 1-par])
    for j in range(M):
        constraints.append(xi_2[j]+xi_3[j] >= cp.kl_div(lbda[j],xi_1[j]) +lbda[j]-xi_1[j])
        constraints.append(xi_2[j] <= cp.geo_mean(cp.vstack([v[j],xi_4[j]]),exp))
    return(constraints)
#### function h(x) = min{x/(1-par),1}
def h_cvar_conj(lbda,v,z,par,constraints):
    M = lbda.shape[0]
    for j in range(M):
        constraints.append(cp.pos(-(1-par)*v[j]+lbda[j])<= z)
    constraints.append(v >= 0)
    return(constraints)
# In[7]:
#### function h(x)=x
def h_lin_conj(lbda,v,z,par,constraints):
    constraints.append(v>=lbda)
    constraints.append(z >= 0)
    return(constraints)

#### The epigraphs h(x)<=0 expressed in cvxpy syntax.

def h_quad_cut(q_b,q,rank,par,constraints):
    N = q.shape[0]
    for i in range(N-1):
        v = (1+par)*cp.sum(q[rank[0:i+1]])-par*cp.sum(q[rank[0:i+1]])**2
        constraints.append(cp.sum(q_b[rank[0:i+1]])-v <= 0)
    return(constraints)

def h_spw_cut(q_b,q,rank,par,constraints):
    N = q.shape[0]
    for i in range(N-1):
        v = 1-cp.power((1-cp.sum(q[rank[0:i+1]])),par)
        constraints.append(cp.sum(q_b[rank[0:i+1]])-v <= 0)
    return(constraints)

def h_dpw_cut(q_b,q,rank,par,constraints):
    N = q.shape[0]
    u1 = cp.Variable(N, nonneg = True)
    u2 = cp.Variable(N, nonneg = True)
    t = cp.Variable(N)
    constraints.append(u1<=1)
    for i in range(N-1):
        constraints.append(cp.sum(q_b[rank[0:i+1]])+t[i]<=0)
        constraints.extend((u2[i]>=-t[i],u1[i]>= 1-cp.sum(q[rank[0:i+1]])))
        constraints.append(cp.power(u1[i],par)+cp.power(u2[i],par) <= 1)
    return(constraints)

def h_pw_cut(q_b,q,rank,par,constraints):
    N = q.shape[0]
    for i in range(N-1):
        constraints.append(cp.sum(q_b[rank[0:i+1]])-cp.power(cp.sum(q[rank[0:i+1]]),par) <= 0)
    return(constraints)

def h_log_cut(q_b,q,rank,par,constraints):
    N = q.shape[0]
    u1 = cp.Variable(N, nonneg = True)
    t = cp.Variable(N)
    constraints.append(u1<=1)
    for i in range(N-1):
        constraints.append(cp.sum(q_b[rank[0:i+1]])+t[i]<=0)
        constraints.append(-u1[i]-cp.entr(u1[i])<= t[i])
        constraints.append(u1[i]-cp.power(cp.sum(q[rank[0:i+1]]),par) <= 0)
    return(constraints)

def h_cvar_cut(q_b, q, rank, par, constraints):
    constraints.append(q_b<= q/(1-par))
    return(constraints)

def h_lin_cut(q_b, q, rank, par, constraints):
    constraints.append(q_b <= q)
    return(constraints)
# In[8]:


##### all h functions evaluation

def h_quad_eva(x,par):
    return((1+par)*x-par*x**2)

def h_spw_eva(x,par):
    return(1-(np.abs(1-x))**par)

def h_dpw_eva(x,par):
    return((1-(np.abs(1-x))**par)**(1/par))

def h_pw_eva(x,par):
    return(x**par)

def h_log_eva(x,par):
    return(x**par*(1-np.log(x**par)))

def h_cvar_eva(x, par):
    return(np.minimum(x/(1-par),1))

def h_lin_eva(x,par):
    return(x)


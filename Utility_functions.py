#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cvxpy as cp


# In[4]:

#### The following uility functions are expressed in the syntax of cvxpy that is suitable for disciplined convex programming
def lin_utility(R,a,W0,par):
    return(W0*(1+R@a))

### power utility function u(x)=(x^{1-par}-1)/(1-par), par>0
def pw_utility(R,a,W0,par):
    N = len(R)
    for i in range(N):
        if i == 0:
            f_obj = (cp.power(W0*(1+(R @ a)[i]),1-par)-1)/(1-par)
        else:
            f_obj = cp.hstack((f_obj,(cp.power(W0*(1+(R @ a)[i]),1-par)-1)/(1-par)))
    return(f_obj)


### exponential utility function u(x)= 1-e^{-x/par}, par>0
def exp_utility(R,a,W0,par):
    N = len(R)
    for i in range(N):
        if i == 0:
            arg = -W0*(1+(R @ a)[i])/par
            f_obj = (1-cp.exp(arg))
        else:
            arg = -W0*(1+(R @ a)[i])/par
            f_obj = cp.hstack((f_obj, 1-cp.exp(arg)))
    return(f_obj)
### same exponential utility function written such that it is suitable for portfolio maximization problem
def exp_utility_pmax(R, r_f, a,W0,par):
    N = len(R)
    for i in range(N):
        if i == 0:
            arg = -(W0*(1+(R@a)[i]+(1-cp.sum(a))*r_f))/par
            f_obj = (1-cp.exp(arg))
        else:
            arg = -(W0*(1+(R@a)[i]+(1-cp.sum(a))*r_f))/par
            f_obj = cp.hstack((f_obj, 1-cp.exp(arg)))
    return(f_obj)


### Same utility functions as above, but soley for value evaluation purposes.

def lin_utility_eva(R,w,W0,par):
    return(W0*(1+R.dot(w)))

def pw_utility_eva(R,w,W0,par):
    N = len(R)
    f_obj = np.zeros(N)
    for i in range(N):
        f_obj[i] = ((W0*(1+(R.dot(w))[i]))**(1-par)-1)/(1-par)
    return(f_obj)

def exp_utility_eva(R,w,W0,par):
    N = len(R)
    f_obj = np.zeros(N)
    for i in range(N):
        #arg = -(R.dot(w))[i]/par
        arg = -(W0*(1+(R.dot(w))[i]))/par
        f_obj[i] = (1-np.exp(arg))
    return(f_obj)

def exp_utility_eva_pmax(R,r_f,w,W0,par):
    N = len(R)
    f_obj = np.zeros(N)
    for i in range(N):
        arg = -(W0*(1+(R.dot(w))[i]+(1-np.sum(w))*r_f))/par
        f_obj[i] = (1-np.exp(arg))
    return(f_obj)

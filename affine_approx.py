#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cvxpy as cp
import seaborn as sns
import mosek
import matplotlib.pyplot as plt
import datetime as date
from datetime import datetime as dt
from dateutil.relativedelta import *
import scipy.stats
from scipy.stats import rankdata


# In[2]:

def argmax_pw(x2,x1,r):
    frac = (x2**r-x1**r)/(x2-x1)*1/r
    return(frac**(1/(r-1)))

def max_error_pw(x2,x1,x,r):
    frac = (x2**r-x1**r)/(x2-x1)
    return(x**r-frac*(x-x1)-x1**r)

def pw (x,r):
    return(x**r)

def argmax_quad(x2,x1,par):
    return((x2+x1)/2)

def max_error_quad(x2,x1,x,par):
    return(-par*x**2+par*(x1+x2)*(x-x1)+par*x1**2)

def quad(x,par):
    return((1+par)*x-par*x**2)

def argmax_sing_power(x2,x1,par):
    breuk = ((1-x1)**par-(1-x2)**par)/(x2-x1)
    return(1-(breuk/par)**(1/(par-1)))

def max_error_singpw(x2,x1,x,par):
    breuk = ((1-x1)**par-(1-x2)**par)/(x2-x1)
    return((1-x1)**par-(1-x)**par-breuk*(x-x1))

def sing_pw(x,par):
    return(1-(1-x)**par)



def piece_affine_eval(x_points, x, h_eval, par):
    value = np.inf
    K = len(x_points)
    for i in range(1,K):
        l = (h_eval(x_points[i],par)-h_eval(x_points[i-1],par))/(x_points[i]-x_points[i-1])
        value2 = l*(x-x_points[i-1])+h_eval(x_points[i-1],par)
        if value > value2:
            value = value2
        else:
            return(value)
    return(value)


# In[3]:


def affine_approx(eps,argmaxfunc, max_error,par):
    x_points = [0]
    x1 ,x2,x2_l,x2_r = [0,1,0,1]
    error = np.inf
    while True:
        x = argmaxfunc(x2,x1,par)
        error = max_error(x2,x1,x,par)
        if np.abs(1-x2)<= 0.00001:
            if error <= eps + 0.00001:
                x_points.append(x2)
                break
        if np.abs(error-eps)<= 0.00001:
            x_points.append(x2)
            x1, x2_l, x2,x2_r = [x2, x1, 1, 1]
        elif error > eps:
            x2_r = x2
            x2 = (x2+x2_l)/2
        elif error < eps:
            x2_l = x2
            x2 = (x2+x2_r)/2
    return(np.array(x_points))


# In[4]:


def makepoints(h_eval,xpoints,par):
    K = len(xpoints)
    slope = []
    b = []
    for i in range(1,K):
        slp = (h_eval(xpoints[i],par)-h_eval(xpoints[i-1],par))/(xpoints[i]-xpoints[i-1])
        slope.append(slp)
        b.append(h_eval(xpoints[i],par)-slp*xpoints[i])
    return(np.array(slope), np.array(b))



def affine_approx_hspw(par,eps):
    x_points = affine_approx(eps,argmax_sing_power,max_error_singpw,par)
    return(x_points)


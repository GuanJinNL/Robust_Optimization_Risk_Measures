# Robust Optimization of Rank-Dependent Models for Uncertain Probabilities
## Introduction
This repository contains the code of the numerical examples that are disussed in the paper Robust Optimization of Rank-Dependent Models for Uncertain Probabilities. The examples that are illustrated are the single-item/multi-item newsvendor problem and a robust portfolio optimization problem. The main codes of each numerical example can be found in its corresponding ipynb file. 

## Dependencies
It is important that the following packages are installed 
+ numpy
+ cvxpy
+ mosek (for this one needs a license, however one can also choose to solve cvxpy with other solvers)
+ Gurobi

## Instructions
Experiments can be run in their corresponding ipynb file. Each of these files are also facilitated with markdown cell that explains the experiment at hand. The codes are provided with comments.
To run the single-item newsvendor experiment, click on the file:
```
Single_Item_Newsvendor.ipynb
```
To run the multi-item newsvendor experiment, click on the file:
```
Multi_Items_Newsvendor.ipynb
```
To run the portfolio optimization experiment, click on the file:
```
Robust_Portfolio.ipynb
```
The .py files are files that contain functions which are needed in the experiments. See the comments in these files for further details.

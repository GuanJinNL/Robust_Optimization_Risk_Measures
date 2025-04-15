# Robust Optimization of Rank-Dependent Models with Uncertain Probabilities

## Introduction
This repository contains the code for the algorithms and numerical examples developed in the paper **Robust Optimization of Rank-Dependent Models with Uncertain Probabilities**. The examples that are analyzed are: (i) a single-item newsvendor problem; (ii) a multi-item newsvendor problem; (iii) a robust portfolio optimization problem with concave distortion function; and (iv) a robust portfolio optimization problem with the inverse S-shaped distortion function of Prelec. See Section 7 of the paper for further details.


## Dependencies
It is important that the following optimization packages are installed: 
+ cvxpy
+ mosek (requires an academic license)
+ Gurobi Version 11.0.3 (requires an academic license)

## Instructions
Experiments can be run in their corresponding ipynb file, where the codes are provided with comments.
To run the single-item newsvendor experiment, click on the file:
```
Single_Item_Newsvendor.ipynb
```
To run the multi-item newsvendor experiment, click on the file:
```
Multi_Items_Newsvendor.ipynb
```
To run the portfolio optimization experiment, import the data file "6_Portfolios_2x3.csv" and click on the file:
```
Robust_Portfolio.ipynb
```
To run the portfolio optimization experiment with inverse S-shaped distortion function (Prelec), click on the file:
```
Prelec_Portfolio.ipynb
```
The .py files are support files that contain functions, which are needed in the experiments. See the comments in these files for further details.

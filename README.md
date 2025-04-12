# Robust Optimization of Rank-Dependent Models with Uncertain Probabilities

## Introduction
This repository contains the code of numerical examples that are discussed in the paper Robust Optimization of Rank-Dependent Models with Uncertain Probabilities. The examples that are analyzed are the single-item/multi-item newsvendor problem and a robust portfolio optimization problem (with concave and inverse S-shaped distortion functions). The main codes of each numerical example can be found in the corresponding ipynb file. 

## Dependencies
It is important that the following packages are installed: 
+ numpy
+ cvxpy
+ mosek (requires an academic license)
+ Gurobi Version 11.0.3 (requires an academic license)

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
To run the portfolio optimization experiment, import the data file "6_Portfolios_2x3.csv" and click on the file:
```
Robust_Portfolio.ipynb
```
To run the portfolio optimization experiment with inverse S-shaped distortion function (Prelec), click on the file:
```
Prelec_Portfolio.ipynb
```
The .py files are support files that contain functions, which are needed in the experiments. See the comments in these files for further details.

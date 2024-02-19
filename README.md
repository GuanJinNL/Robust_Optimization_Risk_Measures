# Robust Optimization of Rank-Dependent Models with Uncertain Probabilities
## Introduction
This repository contains the code of numerical examples that are disussed in the paper Robust Optimization of Rank-Dependent Models with Uncertain Probabilities. The examples that are illustrated are the single-item/multi-item newsvendor problems and a robust portfolio optimization problem (with concanve and non-concave distortion functions). The main codes of each numerical example can be found in their corresponding ipynb file. 

## Dependencies
It is important that the following packages are installed 
+ numpy
+ cvxpy
+ mosek (requires an academic license)
+ Gurobi (requires an academic license)

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
To run the portfolio optimization experiment with non-concave distortion function (Prelec), click on the file:
```
Prelec_Portfolio.ipynb
```
The .py files are support files that contain functions which are needed in the experiments. See the comments in these files for further details.

# Bayesian Analysis of Loan Default Data

## Introduction

This project uses R and Stan to model data about loan defaults related to medical bills.

## File Descriptions

***credit_risk_dataset.csv:*** Data file for the project.

***:*** Code for hierarchical model of parameter for probability of defaulting.

***:*** Code for finding coefficients of logistic regression model with Hamiltonian Monte Carlo.

***:*** Raw Rmarkdown code.

***:*** Code for this project in format that is accessible by GitHub.

## Limitations

Instead of finding prior data for doing Bayesian analysis, I split my original dataset into prior, observed, and test. Datasets for Bayesian analysis should not all come from the same source, so this biased my analysis.

## Methods

First, I explored the parameter for the probability of defaulting. My beta-binomial model adequately represented the data with the posterior distribution. The maximum likelihood estimate of the observed data was captured by the credible interval. Also, Monte Carlo simulations of the obseved and test data captured the true number of defaults. To build off of this model, I made a hierarchical model of the parameter for defaulting on a medical bill. This model considered the three categories for homeownership status: rent, mortgate, and own. The no pooling estimate performed better than the shrinkage and complete pooling estimate. Therefore, I believe that the probability of loan default comes from three separate distributions, and these distributions are determined based on the homeownership status of an individual. However, other confounding variables could also be influencing this relationship.

Next, I built Bayesian logistic regression models to predict whether a loan will default based off of six predictor variables. I used the Metropolis Hastings algorithm and the Hamiltonian Monte Carlo algorithm to add Bayesian elements. Both models had similar coefficients and performed similarly on the train and test data. My data had imbalanced classes, so the models misclassified defaulted loans more often than they misclassified loans that did not default. 

## Improvements

*Hierarchical Model*

- Use ![equation](https://latex.codecogs.com/gif.latex?P(\alpha,\beta)\sim(\alpha&plus;\beta)^{-5/2}) as a prior distribution for ![equation](https://latex.codecogs.com/gif.latex?\alpha) and ![equation](https://latex.codecogs.com/gif.latex?\beta). Sample ![equation](https://latex.codecogs.com/gif.latex?\alpha) and ![equation](https://latex.codecogs.com/gif.latex?\beta) with the Metropolis Hastings algorithm.

*Logistic Regression*

- Use cross validation to tune the Bayesian parameters, like prior standard deviaiton, for both models.
- Apply an imbalanced classification method to the data so the models can learn underlying trends of the default class.

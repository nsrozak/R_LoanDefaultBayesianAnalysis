Bayesian Data Inference
================
Natalie Rozak
6/19/2020

``` r
# global imports
# data preprocessing
library(tidyverse)
library(reshape2)
library(caret)
# data visualization
library(kableExtra)
library(ggplot2)
library(gridExtra)
# models
library(rstan)
library(mvtnorm)
library(coda)
library(ROCR)

# supress scientific notation
options(scipen=999)
```

# Data Preprocessing

The dataset I used came from
<https://www.kaggle.com/laotse/credit-risk-dataset>. This dataset is
about loan defaults.

``` r
# import data
data <- read.csv(
  '~/Documents/GitHub/R_LoanDefaultBayesianAnalysis/credit_risk_dataset.csv')
```

## Properties of Dataset

``` r
# output structure of the data
str(data)
```

    ## 'data.frame':    32581 obs. of  12 variables:
    ##  $ person_age                : int  22 21 25 23 24 21 26 24 24 21 ...
    ##  $ person_income             : int  59000 9600 9600 65500 54400 9900 77100 78956 83000 10000 ...
    ##  $ person_home_ownership     : Factor w/ 4 levels "MORTGAGE","OTHER",..: 4 3 1 4 4 3 4 4 4 3 ...
    ##  $ person_emp_length         : num  123 5 1 4 8 2 8 5 8 6 ...
    ##  $ loan_intent               : Factor w/ 6 levels "DEBTCONSOLIDATION",..: 5 2 4 4 4 6 2 4 5 6 ...
    ##  $ loan_grade                : Factor w/ 7 levels "A","B","C","D",..: 4 2 3 3 3 1 2 2 1 4 ...
    ##  $ loan_amnt                 : int  35000 1000 5500 35000 35000 2500 35000 35000 35000 1600 ...
    ##  $ loan_int_rate             : num  16 11.1 12.9 15.2 14.3 ...
    ##  $ loan_status               : int  1 0 1 1 1 1 1 1 1 1 ...
    ##  $ loan_percent_income       : num  0.59 0.1 0.57 0.53 0.55 0.25 0.45 0.44 0.42 0.16 ...
    ##  $ cb_person_default_on_file : Factor w/ 2 levels "N","Y": 2 1 1 1 2 1 1 1 1 1 ...
    ##  $ cb_person_cred_hist_length: int  3 2 3 2 4 2 3 4 2 3 ...

## Remove Missing Values

``` r
# output number of missing values in each column
kable(t(sapply(data,function(x) sum(is.na(x))))) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=TRUE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:right;">

person\_age

</th>

<th style="text-align:right;">

person\_income

</th>

<th style="text-align:right;">

person\_home\_ownership

</th>

<th style="text-align:right;">

person\_emp\_length

</th>

<th style="text-align:right;">

loan\_intent

</th>

<th style="text-align:right;">

loan\_grade

</th>

<th style="text-align:right;">

loan\_amnt

</th>

<th style="text-align:right;">

loan\_int\_rate

</th>

<th style="text-align:right;">

loan\_status

</th>

<th style="text-align:right;">

loan\_percent\_income

</th>

<th style="text-align:right;">

cb\_person\_default\_on\_file

</th>

<th style="text-align:right;">

cb\_person\_cred\_hist\_length

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

895

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

3116

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

<td style="text-align:right;">

0

</td>

</tr>

</tbody>

</table>

``` r
# remove missing values
data <- data %>% na.omit()
```

## Remove Duplicates

``` r
# output number of duplicates
dim(data[duplicated(data),])
```

    ## [1] 137  12

``` r
# remove duplicates
data <- data[!duplicated(data),]
```

## Select Rows and Preliminary Columns for Analysis

``` r
# remove most categorical variables to make hierarhcical model simploer
data <- data %>% subset(select = -c(loan_grade,cb_person_default_on_file))
# obtain rows corresponding to medical
data <- data %>% filter(loan_intent=='MEDICAL')
# remove rows corresponding to `other`
data <- data %>% filter(person_home_ownership != 'OTHER')
# correct factor levels
data$person_home_ownership <- factor(data$person_home_ownership)
# remove variable
data <- data %>% subset(select=-c(loan_intent))
```

## Split Data into Prior, Observed, and Test

``` r
# obtain observed size
n_observed <- floor(0.5*nrow(data))
# obtain prior indices
set.seed(897)
observed_ind <- sample(seq_len(nrow(data)),size=n_observed)
# obtain prior dataset
observed <- data[observed_ind,]
other <- data[-observed_ind,]
# obtain prior size
n_prior <- floor(0.5*nrow(other))
# obtain prior indicies
set.seed(981)
prior_ind <- sample(seq_len(nrow(other)),size=n_prior)
# obtain prior and test datasets
prior <- other[prior_ind,]
test <- other[-prior_ind,]
```

## Variable Selection

``` r
# function for stating whether a hypothesis test is statistically significant
statistically_significant <- function(pvalue){
  if (pvalue < 0.05){
    cat('Statistically significant\n')
  } else {
    cat('Not statistically significant\n')
  }
}
```

### Chi-Squared Test for Categorical Predictors

The null hypothesis for the chi-squared test is that home ownership type
and loan status are independent. These variables are not independed in
the alternative hypothesis for the chi-squared test.

The test statistic is ![\\chi^2 =
\\sum\_{i=1}^3\\sum\_{j=1}^2\\frac{(x\_{ij}-e\_{ij})^2}{e\_{ij}}](https://ibm.codecogs.com/png.latex?%5Cchi%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5E3%5Csum_%7Bj%3D1%7D%5E2%5Cfrac%7B%28x_%7Bij%7D-e_%7Bij%7D%29%5E2%7D%7Be_%7Bij%7D%7D
"\\chi^2 = \\sum_{i=1}^3\\sum_{j=1}^2\\frac{(x_{ij}-e_{ij})^2}{e_{ij}}")
where
![e\_{ij}=\\frac{r\_ic\_j}{n}](https://ibm.codecogs.com/png.latex?e_%7Bij%7D%3D%5Cfrac%7Br_ic_j%7D%7Bn%7D
"e_{ij}=\\frac{r_ic_j}{n}"), and there are two degrees of freedom for
this contingency table. Under the null hypothesis, ![\\chi^2 \\sim
\\chi^2\_{(2)(1)}](https://ibm.codecogs.com/png.latex?%5Cchi%5E2%20%5Csim%20%5Cchi%5E2_%7B%282%29%281%29%7D
"\\chi^2 \\sim \\chi^2_{(2)(1)}").

``` r
# outout levels in person_home_ownership
kable(table(observed$person_home_ownership,observed$loan_status),
      col.names=c('No Default','Default'))%>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

No Default

</th>

<th style="text-align:right;">

Default

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

MORTGAGE

</td>

<td style="text-align:right;">

724

</td>

<td style="text-align:right;">

223

</td>

</tr>

<tr>

<td style="text-align:left;">

OWN

</td>

<td style="text-align:right;">

173

</td>

<td style="text-align:right;">

12

</td>

</tr>

<tr>

<td style="text-align:left;">

RENT

</td>

<td style="text-align:right;">

1004

</td>

<td style="text-align:right;">

491

</td>

</tr>

</tbody>

</table>

``` r
# chi-squared test for person_home_ownership
homeowner_chisq <- table(observed$person_home_ownership,observed$loan_status)
homeowner_test <- chisq.test(homeowner_chisq)
# output results
statistically_significant(homeowner_test$p.value)
```

    ## Statistically significant

Since my p-value is less than alpha=0.05, I have statistically
significant evidence to reject my null hypothesis. Homeowner type and
loan status are not independent, so I will use this variable in the
logistic regression model.

### Mann-Whitney U Test for Continuous Predictors

***Visualize Variables***

``` r
# function for creating a violin plot
violin_plot <- function(x,y,lab){
  ggplot() +
    geom_violin(aes_string(x=x,y=y,color=x,fill=x),show.legend=FALSE) +
    labs(title='Violin Plot') + ylab(lab) + xlab('Loan Default') + 
    scale_color_manual(name='Legend',
                       values=c('0'='deepskyblue2',
                                '1'='darkorange2'),
                       labels=c('No Default','Default'))+
    scale_fill_manual(name='Legend',
                       values=c('0'='deepskyblue2',
                                '1'='darkorange2'),
                       labels=c('No Default','Default'))+
    theme(plot.title=element_text(face='bold'))
}
```

``` r
# lists used in for loop
num_cols <- c('person_age','person_income','person_emp_length','loan_amnt',
              'loan_int_rate','loan_percent_income','cb_person_cred_hist_length')
y_name <- c('Age','Income','Length of Employment','Loan Amount','Interest Rate',
            'Loan Percent of Income','Credit History Length')
plots <- list()
# create violin plot for each variable
for (i in 1:length(num_cols)){
  plots[[i]] <-violin_plot(factor(observed$loan_status),observed[[num_cols[i]]],y_name[i])
}
# plot grid
do.call(grid.arrange,c(plots,ncol=3))
```

<img src="BayesianDataInference_files/figure-gfm/violin plots-1.png" style="display: block; margin: auto;" />

The violin plots show that age, income, length of employment, and credit
history have fairly similar distributions.

***Run Hypothesis Test***

The null hypothesis for the Mann-Whitney U test is that the ranks are
uniformly distributed and the distributions of default and no default
values are identical. The alternative hypothesis is that these
distributions are not identical.

The Mann-Whitney test statistic is
![U=W-\\frac{m(m+1)}{2}](https://ibm.codecogs.com/png.latex?U%3DW-%5Cfrac%7Bm%28m%2B1%29%7D%7B2%7D
"U=W-\\frac{m(m+1)}{2}") where ![W](https://ibm.codecogs.com/png.latex?W
"W") is the sum of the ranks from sample
![X](https://ibm.codecogs.com/png.latex?X "X") and
![m](https://ibm.codecogs.com/png.latex?m "m") is the number of
observations in ![X](https://ibm.codecogs.com/png.latex?X "X").

``` r
# Mann Whitney U test for age
age_test <- wilcox.test(observed$person_age~observed$loan_status)
# output results
statistically_significant(age_test$p.value)
```

    ## Not statistically significant

Since my p-value is greater than alpha=0.05, I do not have statistically
significant evidence to reject my null hypothesis. Loan status is not
impacted by the age of the person.

``` r
# Mann Whitney U test for income
income_test <- wilcox.test(observed$person_income~observed$loan_status)
# output results
statistically_significant(income_test$p.value)
```

    ## Statistically significant

Since my p-value is less than alpha=0.05, I have statistically
significant evidence to reject my null hypothesis. Loan status is
impacted by the income of the person.

``` r
# Mann Whitney U test for employment length
employment_test <- wilcox.test(observed$person_emp_length~observed$loan_status)
# output results
statistically_significant(employment_test$p.value)
```

    ## Not statistically significant

Since my p-value is greater than alpha=0.05, I do not have statistically
significant evidence to reject my null hypothesis. Loan status is not
impacted by amount of time that a person was employed.

``` r
# Mann Whitney U test for loan amount
amount_test <- wilcox.test(observed$loan_amnt~observed$loan_status)
# output results
statistically_significant(amount_test$p.value)
```

    ## Statistically significant

Since my p-value is less than alpha=0.05, I have statistically
significant evidence to reject my null hypothesis. Loan status is
impacted by the loan amount.

``` r
# Mann Whitney U test for interest
interest_test <- wilcox.test(observed$loan_int_rate~observed$loan_status)
# output results
statistically_significant(interest_test$p.value)
```

    ## Statistically significant

Since my p-value is less than alpha=0.05, I have statistically
significant evidence to reject my null hypothesis. Loan status is
impacted by the loan interest rate.

``` r
# Mann Whitney U test for percent income
percent_income_test <- wilcox.test(observed$loan_percent_income~observed$loan_status)
# output results
statistically_significant(percent_income_test$p.value)
```

    ## Statistically significant

Since my p-value is less than alpha=0.05, I have statistically
significant evidence to reject my null hypothesis. Loan status is
impacted by the percentage of a person’s income that a loan equates to.

``` r
# Mann Whitney U test for credit history
credit_history_test <- wilcox.test(
  observed$cb_person_cred_hist_length~observed$loan_status)
# output results
statistically_significant(credit_history_test$p.value)
```

    ## Not statistically significant

Since my p-value is greater than alpha=0.05, I do not have statistically
significant evidence to reject my null hypothesis. Loan status is not
impacted by the credit history of the person.

``` r
# remove columns that are not statistically significant predictors
prior <- prior %>%
  subset(select=-c(person_age,person_emp_length,cb_person_cred_hist_length))
observed <- observed %>%
  subset(select=-c(person_age,person_emp_length,cb_person_cred_hist_length))
test <- test %>% 
  subset(select=-c(person_age,person_emp_length,cb_person_cred_hist_length))
```

# Exploration of `loan_status` Variable

## Levels of `loan_status`

***Prior Dataset***

``` r
# outout levels in target
kable(table(prior$loan_status), col.names=c('Level','Count')) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

Level

</th>

<th style="text-align:right;">

Count

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

980

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

334

</td>

</tr>

</tbody>

</table>

***Observed Dataset***

``` r
# outout levels in target
kable(table(observed$loan_status), col.names=c('Level','Count')) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

Level

</th>

<th style="text-align:right;">

Count

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

1901

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

726

</td>

</tr>

</tbody>

</table>

## Beta-Binomial Model for `loan_status`

``` r
# x values for functions
x <- seq(0.21,0.32,length=1000)
# function for plotting bayes model
bayes_model_plot <- function(model){
  ggplot(model) + 
    geom_line(aes(x=x,y=Density,color=Model)) +
    labs(title='Bayesian Model') + ylab('Density') +
    scale_color_manual(name='Models',
                       values=c('Prior'='deepskyblue2',
                                'Observed'='darkorange2',
                                'Posterior'='maroon3'),
                       labels=c('Prior','Observed','Posterior')) +
    theme(plot.title=element_text(face='bold'))
}
```

``` r
# obtain parameters
n <- nrow(observed)
y <- sum(observed$loan_status)
# create parameters for sampling distribution
observed_dist <- dbeta(x,y+1,n-y+1)
```

Since the beta pdf and binomial pdf are proportional, I decided to use
the beta pdf to describe the observed distribution. Doing this makes the
graph more readable.

### Model Building

Apriori, ![\\theta\\sim
Beta(334,980)](https://ibm.codecogs.com/png.latex?%5Ctheta%5Csim%20Beta%28334%2C980%29
"\\theta\\sim Beta(334,980)") with ![P(\\theta)\\propto
\\theta^{333}(1-\\theta)^{980}](https://ibm.codecogs.com/png.latex?P%28%5Ctheta%29%5Cpropto%20%5Ctheta%5E%7B333%7D%281-%5Ctheta%29%5E%7B980%7D
"P(\\theta)\\propto \\theta^{333}(1-\\theta)^{980}"). Observe
![Y|\\theta\\sim
Bin(2627,0.2763)](https://ibm.codecogs.com/png.latex?Y%7C%5Ctheta%5Csim%20Bin%282627%2C0.2763%29
"Y|\\theta\\sim Bin(2627,0.2763)") with ![P(Y|\\theta)\\propto
\\theta^{726}(1-\\theta)^{1901}](https://ibm.codecogs.com/png.latex?P%28Y%7C%5Ctheta%29%5Cpropto%20%5Ctheta%5E%7B726%7D%281-%5Ctheta%29%5E%7B1901%7D
"P(Y|\\theta)\\propto \\theta^{726}(1-\\theta)^{1901}"). Then the
posterior distribution is ![\\theta|726\\sim
Beta(1060,2881)](https://ibm.codecogs.com/png.latex?%5Ctheta%7C726%5Csim%20Beta%281060%2C2881%29
"\\theta|726\\sim Beta(1060,2881)") with ![P(\\theta|726)\\propto
\\theta^{1059}(1-\\theta)^{2880}](https://ibm.codecogs.com/png.latex?P%28%5Ctheta%7C726%29%5Cpropto%20%5Ctheta%5E%7B1059%7D%281-%5Ctheta%29%5E%7B2880%7D
"P(\\theta|726)\\propto \\theta^{1059}(1-\\theta)^{2880}").

``` r
# obtain parameters
alpha <- sum(prior$loan_status)
beta <- nrow(prior)-alpha
a_post <- alpha+y
b_post <- beta+n-y
# create the prior and posterior distributions
prior_dist <- dbeta(x,alpha,beta)
posterior_dist <- dbeta(x,a_post,b_post)
# create model name vectors
prior_name <- rep('Prior',length(x))
observed_name <- rep('Observed',length(x))
posterior_name <- rep('Posterior',length(x))
# create data frames
prior_df <- data.frame(prior_name,prior_dist,x)
prior_df <- prior_df %>% plyr::rename(c('prior_name'='Model','prior_dist'='Density'))
observed_df <- data.frame(observed_name,observed_dist,x)
observed_df <- observed_df %>%
  plyr::rename(c('observed_name'='Model','observed_dist'='Density','x'))
posterior_df <- data.frame(posterior_name,posterior_dist,x)
posterior_df <- posterior_df %>%
  plyr::rename(c('posterior_name'='Model','posterior_dist'='Density'))
conjugate <- rbind(prior_df,observed_df,posterior_df)
# create graph
bayes_model_plot(conjugate)
```

<img src="BayesianDataInference_files/figure-gfm/conjugate graph-1.png" style="display: block; margin: auto;" />

### Posterior Mean

![E(\\theta|726)=w\\hat{\\theta}\_{MLE}+(1-w)\\hat{\\theta}\_{prior\\:mean}=\\frac{2627}{3941}\\frac{726}{2627}+\\frac{1314}{3941}\\frac{334}{1314}=0.2689](https://ibm.codecogs.com/png.latex?E%28%5Ctheta%7C726%29%3Dw%5Chat%7B%5Ctheta%7D_%7BMLE%7D%2B%281-w%29%5Chat%7B%5Ctheta%7D_%7Bprior%5C%3Amean%7D%3D%5Cfrac%7B2627%7D%7B3941%7D%5Cfrac%7B726%7D%7B2627%7D%2B%5Cfrac%7B1314%7D%7B3941%7D%5Cfrac%7B334%7D%7B1314%7D%3D0.2689
"E(\\theta|726)=w\\hat{\\theta}_{MLE}+(1-w)\\hat{\\theta}_{prior\\:mean}=\\frac{2627}{3941}\\frac{726}{2627}+\\frac{1314}{3941}\\frac{334}{1314}=0.2689").

### Bias and Variance

The bias of the model is
![E(w\\bar{Y}+(1-w)\\theta\_0-\\theta)=(1-w)(\\theta\_0-\\theta)=0.3334(0.2541-\\theta)](https://ibm.codecogs.com/png.latex?E%28w%5Cbar%7BY%7D%2B%281-w%29%5Ctheta_0-%5Ctheta%29%3D%281-w%29%28%5Ctheta_0-%5Ctheta%29%3D0.3334%280.2541-%5Ctheta%29
"E(w\\bar{Y}+(1-w)\\theta_0-\\theta)=(1-w)(\\theta_0-\\theta)=0.3334(0.2541-\\theta)").
The variance of the model is
![Var(w\\bar{Y}+(1-w)\\theta\_0)=w^2Var(\\bar{Y})=0.1111Var(\\bar{Y})](https://ibm.codecogs.com/png.latex?Var%28w%5Cbar%7BY%7D%2B%281-w%29%5Ctheta_0%29%3Dw%5E2Var%28%5Cbar%7BY%7D%29%3D0.1111Var%28%5Cbar%7BY%7D%29
"Var(w\\bar{Y}+(1-w)\\theta_0)=w^2Var(\\bar{Y})=0.1111Var(\\bar{Y})").
The bias increases and the variance decreases when using the Bayesian
estimate instead of the maximum likelihood estimate.

### Posterior Predictive Distribution

The posterior predictive distribution is a Beta-Binomial model with
probability density
![P(\\tilde{y}|y)=\\frac{\\Gamma(n+1)}{\\Gamma(\\tilde{y}+1)\\Gamma(n-\\tilde{y}+1)}\\frac{\\Gamma(3941)}{\\Gamma(1060)\\Gamma(2881)}\\frac{\\Gamma(\\tilde{y}+1060)\\Gamma(n-\\tilde{y}+2881)}{\\Gamma(n+3941)}](https://ibm.codecogs.com/png.latex?P%28%5Ctilde%7By%7D%7Cy%29%3D%5Cfrac%7B%5CGamma%28n%2B1%29%7D%7B%5CGamma%28%5Ctilde%7By%7D%2B1%29%5CGamma%28n-%5Ctilde%7By%7D%2B1%29%7D%5Cfrac%7B%5CGamma%283941%29%7D%7B%5CGamma%281060%29%5CGamma%282881%29%7D%5Cfrac%7B%5CGamma%28%5Ctilde%7By%7D%2B1060%29%5CGamma%28n-%5Ctilde%7By%7D%2B2881%29%7D%7B%5CGamma%28n%2B3941%29%7D
"P(\\tilde{y}|y)=\\frac{\\Gamma(n+1)}{\\Gamma(\\tilde{y}+1)\\Gamma(n-\\tilde{y}+1)}\\frac{\\Gamma(3941)}{\\Gamma(1060)\\Gamma(2881)}\\frac{\\Gamma(\\tilde{y}+1060)\\Gamma(n-\\tilde{y}+2881)}{\\Gamma(n+3941)}").

### Monte Carlo Simulation

Run a Monte Carlo simulation with test statistic ![t =
E(y^{(s)})](https://ibm.codecogs.com/png.latex?t%20%3D%20E%28y%5E%7B%28s%29%7D%29
"t = E(y^{(s)})") from sampling model with
![\\theta^{(s)}](https://ibm.codecogs.com/png.latex?%5Ctheta%5E%7B%28s%29%7D
"\\theta^{(s)}") as a parameter.

``` r
# monte carlo simulation function
monte_carlo <- function(size){
  T_ytilde <- c()
  for (s in 1:1000){
    # obtain theta_s from posterior distribution
    theta_s <- rbeta(1,a_post,b_post)
    # obtain 10 y values from sampling model with parameter theta_s
    y_s <- rbinom(10,size,theta_s)
    # calculate the test statistic
    t_s <- mean(y_s)
    T_ytilde <- c(T_ytilde,t_s)
  }
  return(T_ytilde)
}
# monte carlo simulation graphs
monte_carlo_plot <- function(test_stat, real){
  ggplot() +
    geom_histogram(aes(x=test_stat,y=..density..),bins=20,alpha=0.5,fill='deepskyblue2') +
    geom_density(aes(x=test_stat),alpha=0.3,color='deepskyblue2',fill='deepskyblue2') + 
    geom_segment(
      aes(x=real,y=0,xend=real,yend=Inf),color='darkorange2',linetype='dashed') +
    labs(title='Monte Carlo Simulation') + 
    xlab('Number of Defaults') + ylab('Density') +
    theme(plot.title=element_text(face='bold'))
}
```

***Observed Data***

``` r
# obtain test statistic for observed and simulated data
T_yobs <- sum(observed$loan_status)
T_ytilde <- monte_carlo(n)
# create plot
monte_carlo_plot(T_ytilde,T_yobs)
```

<img src="BayesianDataInference_files/figure-gfm/monte carlo on observed data-1.png" style="display: block; margin: auto;" />

The observed number of defaults lies on the histogram for the Monte
Carlo simulated number of defaults.

***Test Data***

``` r
# obtain test statistic for observed and simulated data
T_ytest <- sum(test$loan_status)
T_ytilde <- monte_carlo(nrow(test))
# create plot
monte_carlo_plot(T_ytilde,T_ytest)
```

<img src="BayesianDataInference_files/figure-gfm/monte carlo on test data-1.png" style="display: block; margin: auto;" />

The test number of defaults lies on the histogram for the Monte Carlo
simulated number of defaults.

### Posterior Credible Interval

``` r
# function for plotting the credible interval
cred_int_plot <- function(model,lower,upper){
  ggplot() +
    geom_line(aes(x,model),color='deepskyblue2') +
    geom_area(mapping=aes(x=ifelse(x>lower & x<upper,x,0),y=model),
      fill='deepskyblue2', alpha=0.3) + 
    xlim(0.23,0.31) +
    labs(title='Credible Interval') + 
    xlab('Theta') + ylab('Density') +
    theme(plot.title=element_text(face='bold'))
}
```

``` r
# MLE for observed data
p <- sum(observed$loan_status)/nrow(observed)
# obtain bounds
bounds <- qbeta(c(0.025,0.975),a_post,b_post)
# print bounds
cat('Lower bound: ',bounds[1],'\n')
```

    ## Lower bound:  0.2552371

``` r
cat('Upper bound: ',bounds[2],'\n')
```

    ## Upper bound:  0.2829195

There is a 95% chance that
![\\theta](https://ibm.codecogs.com/png.latex?%5Ctheta "\\theta") lies
between 0.2552 and 0.2829.

``` r
# plot credible interval with MLE from observed data
cred_int_plot(posterior_dist,bounds[1],bounds[2]) + 
  geom_segment(aes(x=p,y=0,xend=p,yend=Inf),color='darkorange2',linetype='dashed')
```

<img src="BayesianDataInference_files/figure-gfm/plot credbile interval-1.png" style="display: block; margin: auto;" />

The maximum likelihood estimate from the observed data lies in the
credible interval for
![\\theta](https://ibm.codecogs.com/png.latex?%5Ctheta "\\theta").

## Hierarchical Model for `loan_status`

``` r
# observed data
h_model_obs <- observed %>% 
  group_by(person_home_ownership) %>% 
  summarise(y=sum(loan_status),n=n())
# test data
h_test <- test %>% 
  group_by(person_home_ownership) %>% 
  summarise(y=sum(loan_status),n=n()) %>%
  mutate(percent=y/n)
# add test data to h_model
h_model_obs$test <- h_test$percent
# obtain values
y <- h_model_obs$y
n <- h_model_obs$n
J <- nrow(h_model_obs)
```

### No Pooling

The no pooling estimate for the parameter of each group is
![\\hat{\\theta}\_i^{(mle)}=yi](https://ibm.codecogs.com/png.latex?%5Chat%7B%5Ctheta%7D_i%5E%7B%28mle%29%7D%3Dyi
"\\hat{\\theta}_i^{(mle)}=yi").

``` r
# no pooling estimate
no_pool <- y/n
# output the no pooling estimate for each group
for (i in 1:length(h_model_obs$person_home_ownership)){
  cat('No pooling estimate for ',
      as.vector(h_model_obs$person_home_ownership)[i],' : ',no_pool[i],'\n')
}
```

    ## No pooling estimate for  MORTGAGE  :  0.2354805 
    ## No pooling estimate for  OWN  :  0.06486486 
    ## No pooling estimate for  RENT  :  0.3284281

### Complete Pooling

The complete pooling estimate for the parameter of each group is
![\\hat{\\theta}\_i^{(pool)}=\\frac{\\sum\_jy\_j}{\\sum\_jn\_j}](https://ibm.codecogs.com/png.latex?%5Chat%7B%5Ctheta%7D_i%5E%7B%28pool%29%7D%3D%5Cfrac%7B%5Csum_jy_j%7D%7B%5Csum_jn_j%7D
"\\hat{\\theta}_i^{(pool)}=\\frac{\\sum_jy_j}{\\sum_jn_j}").

``` r
# complete pooling estimate
complete_pool <- sum(y)/sum(n)
# output the complete pooling estimate
cat('Complete pooling estimate: ',complete_pool)
```

    ## Complete pooling estimate:  0.2763609

### Shrinkage Model

The shrinkage estimate for the parameter of each group is
![\\hat{\\theta}\_i^{(shrink)}=w\\hat{\\theta}\_i^{(mle)}+(1-w)\\hat{\\theta}\_i^{(pool)}](https://ibm.codecogs.com/png.latex?%5Chat%7B%5Ctheta%7D_i%5E%7B%28shrink%29%7D%3Dw%5Chat%7B%5Ctheta%7D_i%5E%7B%28mle%29%7D%2B%281-w%29%5Chat%7B%5Ctheta%7D_i%5E%7B%28pool%29%7D
"\\hat{\\theta}_i^{(shrink)}=w\\hat{\\theta}_i^{(mle)}+(1-w)\\hat{\\theta}_i^{(pool)}").

The sampling model is ![Y\_i\\sim
Bin(n\_i,\\theta\_i)](https://ibm.codecogs.com/png.latex?Y_i%5Csim%20Bin%28n_i%2C%5Ctheta_i%29
"Y_i\\sim Bin(n_i,\\theta_i)"). The parameter for each group is
![\\theta\_i \\sim
Beta(\\alpha,\\beta)](https://ibm.codecogs.com/png.latex?%5Ctheta_i%20%5Csim%20Beta%28%5Calpha%2C%5Cbeta%29
"\\theta_i \\sim Beta(\\alpha,\\beta)"). Lastly, ![\\alpha,\\beta\\sim
P(\\alpha,\\beta)](https://ibm.codecogs.com/png.latex?%5Calpha%2C%5Cbeta%5Csim%20P%28%5Calpha%2C%5Cbeta%29
"\\alpha,\\beta\\sim P(\\alpha,\\beta)"). The full model is
![P(y,\\theta,\\alpha,\\beta)=\\prod\_{i=1}^N{n\_i \\choose
y\_i}\\theta\_i^{y\_i}(1-\\theta\_i)^{n\_i-y\_i}\\prod\_{i=1}^N\\frac{\\Gamma(\\alpha+\\beta)}{\\Gamma(\\alpha)\\Gamma(\\beta)}\\theta\_i^{\\alpha-1}(1-\\theta\_i)^{\\beta-1}P(\\alpha,\\beta)](https://ibm.codecogs.com/png.latex?P%28y%2C%5Ctheta%2C%5Calpha%2C%5Cbeta%29%3D%5Cprod_%7Bi%3D1%7D%5EN%7Bn_i%20%5Cchoose%20y_i%7D%5Ctheta_i%5E%7By_i%7D%281-%5Ctheta_i%29%5E%7Bn_i-y_i%7D%5Cprod_%7Bi%3D1%7D%5EN%5Cfrac%7B%5CGamma%28%5Calpha%2B%5Cbeta%29%7D%7B%5CGamma%28%5Calpha%29%5CGamma%28%5Cbeta%29%7D%5Ctheta_i%5E%7B%5Calpha-1%7D%281-%5Ctheta_i%29%5E%7B%5Cbeta-1%7DP%28%5Calpha%2C%5Cbeta%29
"P(y,\\theta,\\alpha,\\beta)=\\prod_{i=1}^N{n_i \\choose y_i}\\theta_i^{y_i}(1-\\theta_i)^{n_i-y_i}\\prod_{i=1}^N\\frac{\\Gamma(\\alpha+\\beta)}{\\Gamma(\\alpha)\\Gamma(\\beta)}\\theta_i^{\\alpha-1}(1-\\theta_i)^{\\beta-1}P(\\alpha,\\beta)").

#### Prior Data

``` r
# function for counting number of positive in a bootstrap sample
bootstrap <- function(vector,num_samp){
  num_pos <- c()
  for (i in 1:num_samp){
    set.seed(187+i)
    samp <- sample(vector,nrow(prior),replace=TRUE)
    num_pos <- c(num_pos,sum(samp))
  }
  return(num_pos)
}
# function for plotting bootstrapped sample
bootstrap_plot <- function(samples, mu){
  ggplot() +
    geom_histogram(aes(x=samples,y=..density..),bins=20,alpha=0.5,fill='deepskyblue2') +
    geom_density(aes(x=samples),alpha=0.3,color='deepskyblue2',fill='deepskyblue2') + 
    geom_segment(aes(x=mu,y=0,xend=mu,yend=Inf),color='darkorange2',linetype='dashed') +
    labs(title='Bootstrap Sample') + 
    xlab('Number of Observations') + ylab('Density') +
    theme(plot.title=element_text(face='bold'))
}
```

``` r
# obtain samples of sample size
pos_samples <- bootstrap(prior$loan_status,1000)
p <- sum(prior$loan_status)/nrow(prior)
samples <- pos_samples/p
# output mean and standard deviation
cat("Mean of samples: ",mean(samples),'\n')
```

    ## Mean of samples:  1314.85

``` r
cat("Standard deviation of samples: ",sd(samples),'\n')
```

    ## Standard deviation of samples:  59.11249

``` r
# plot bootstrap samples
bootstrap_plot(samples,mean(samples))
```

<img src="BayesianDataInference_files/figure-gfm/histogram of prior distribution-1.png" style="display: block; margin: auto;" />

To determine the prior distribution for
![\\alpha](https://ibm.codecogs.com/png.latex?%5Calpha "\\alpha") and
![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta"), I
considered ![\\mu=\\frac{\\alpha}{\\alpha+\\beta}\\sim
Beta(860,2293)](https://ibm.codecogs.com/png.latex?%5Cmu%3D%5Cfrac%7B%5Calpha%7D%7B%5Calpha%2B%5Cbeta%7D%5Csim%20Beta%28860%2C2293%29
"\\mu=\\frac{\\alpha}{\\alpha+\\beta}\\sim Beta(860,2293)"). Next, I
created a distribution for the prior sample size ![\\eta \\sim
N(\\mu\_0,\\sigma\_0)](https://ibm.codecogs.com/png.latex?%5Ceta%20%5Csim%20N%28%5Cmu_0%2C%5Csigma_0%29
"\\eta \\sim N(\\mu_0,\\sigma_0)"). Then ![\\alpha=\\eta\\times
\\mu](https://ibm.codecogs.com/png.latex?%5Calpha%3D%5Ceta%5Ctimes%20%5Cmu
"\\alpha=\\eta\\times \\mu") and
![\\beta=\\eta\\times(1-\\mu)](https://ibm.codecogs.com/png.latex?%5Cbeta%3D%5Ceta%5Ctimes%281-%5Cmu%29
"\\beta=\\eta\\times(1-\\mu)").

#### Model Building

``` r
# apriori parameters
m <- mean(samples)
std <- sd(samples)
# attach stan file
h_model <- stan_model('~/Documents/MachineLearning/Bayes/loan_status.stan')
```

    ## Trying to compile a simple C file

    ## Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    ## clang -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -D_REENTRANT  -DBOOST_DISABLE_ASSERTS -DBOOST_PENDING_INTEGER_LOG2_HPP -include stan/math/prim/mat/fun/Eigen.hpp   -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -I/usr/local/include  -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:88:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:1: error: unknown type name 'namespace'
    ## namespace Eigen {
    ## ^
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:16: error: expected ';' after top level declarator
    ## namespace Eigen {
    ##                ^
    ##                ;
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    ## #include <complex>
    ##          ^~~~~~~~~
    ## 3 errors generated.
    ## make: *** [foo.o] Error 1

``` r
# fit posterior distribution
set.seed(987)
stan_fit <- rstan::sampling(
  h_model, data=list(J=J,y=y,n=n,a=alpha,b=beta,m=m,std=std),refresh=0)
# samples from posterior distribution
h_model_samples <- rstan::extract(stan_fit)
```

``` r
# obtain values
shrinkage <- colMeans(h_model_samples$theta)
# output shrinkage estimate for each group
for (i in 1:length(h_model_obs$person_home_ownership)){
  cat('Prior data shrinkage estimate for ',
      as.vector(h_model_obs$person_home_ownership)[i],
      ' : ',shrinkage[i],'\n')
}
```

    ## Prior data shrinkage estimate for  MORTGAGE  :  0.2478685 
    ## Prior data shrinkage estimate for  OWN  :  0.232905 
    ## Prior data shrinkage estimate for  RENT  :  0.2952806

#### Results

``` r
# obtain data frame
thetas_df <- data.frame(h_model_samples$theta)
thetas_df <- reshape2::melt(thetas_df)
# create histogram
ggplot(thetas_df,aes(x=value,fill=variable)) +
  geom_histogram(aes(y=..density..),bins=20,alpha=0.5,position='identity') +
  geom_density(aes(color=variable),alpha=0.3) +
  scale_color_manual(name='Theta',
                       values=c('X1'='deepskyblue2',
                                'X2'='darkorange2',
                                'X3'='maroon3'),
                       labels=c('Mortgage','Own','Rent')) +
  scale_fill_manual(name='Theta',
                       values=c('X1'='deepskyblue2',
                                'X2'='darkorange2',
                                'X3'='maroon3'),
                       labels=c('Mortgage','Own','Rent')) +
  labs(title='Histogram of Hierarchical Thetas') + 
  xlab('Theta') + ylab('Density') +
  theme(plot.title=element_text(face='bold'))
```

<img src="BayesianDataInference_files/figure-gfm/theta histogram-1.png" style="display: block; margin: auto;" />

``` r
# shrinkage plot
ggplot() +
  geom_segment(aes(x=no_pool,xend=shrinkage,y=1,yend=0,
                   color='Shrinkage Estimates')) +
  geom_point(aes(x=no_pool,y=1),color='deepskyblue2') + 
  geom_point(aes(x=shrinkage,y=0),color='deepskyblue2') +
  geom_segment(aes(x=mean(shrinkage),xend=mean(shrinkage),y=0,yend=1,
                   color='Shrinkage Mean')) + 
  geom_point(aes(x=mean(shrinkage),y=1),color='darkorange2') + 
  geom_point(aes(x=mean(shrinkage),y=0),color='darkorange2') +
  geom_segment(aes(x=complete_pool,xend=complete_pool,y=0,yend=1,
                   color='Complete Pooling Mean')) +
  geom_point(aes(x=complete_pool,y=1),color='maroon3') + 
  geom_point(aes(x=complete_pool,y=0),color='maroon3') +
  labs(title='Shrinkage Plot') + 
  xlab('Theta') + ylab('') +
  scale_y_continuous(breaks=c(0, 1),labels=c("Bayes", "MLE"),limits=c(0,1)) +
  scale_color_manual(name='Legend',values=c('Shrinkage Estimates'='deepskyblue2',
                                            'Shrinkage Mean'='darkorange2',
                                            'Complete Pooling Mean'='maroon3')) +
  theme(plot.title=element_text(face='bold'))
```

<img src="BayesianDataInference_files/figure-gfm/shrinkage plot-1.png" style="display: block; margin: auto;" />

The shrinkage estimates are much closer to the shrinkage mean and
complete pooling mean than they are to the MLE.

### Compare Models

``` r
# rmse function
rmse <- function(y,y_hat){
  return(sqrt(mean((y-y_hat)^2)))
}
```

``` r
# output results
cat('No pooling RMSE: ',rmse(h_model_obs$test,no_pool),'\n')
```

    ## No pooling RMSE:  0.03469015

``` r
cat('Complete pooling RMSE: ',rmse(h_model_obs$test,complete_pool),'\n')
```

    ## Complete pooling RMSE:  0.1093445

``` r
cat('Shrinkage RMSE',rmse(h_model_obs$test,colMeans(h_model_samples$theta)),'\n')
```

    ## Shrinkage RMSE 0.07787976

The no pooling estimate has the lowest RMSE, meaning it is the best
model. Therefore, differences in home ownership status significantly
impact the probability of a loan defaulting. It seems likely that the
loan default parameter comes from a different distribution for each home
ownership status level.

# Bayesian Logistic Regression

Apriori, assume ![\\boldsymbol{\\beta} \\propto
C](https://ibm.codecogs.com/png.latex?%5Cboldsymbol%7B%5Cbeta%7D%20%5Cpropto%20C
"\\boldsymbol{\\beta} \\propto C"). The sampling model is ![Y\_i\\sim
Bin(n,\\theta)](https://ibm.codecogs.com/png.latex?Y_i%5Csim%20Bin%28n%2C%5Ctheta%29
"Y_i\\sim Bin(n,\\theta)") where ![\\theta=\\frac{e^{\\boldsymbol{\\beta
X}}}{1+ e^{\\boldsymbol{\\beta
X}}}](https://ibm.codecogs.com/png.latex?%5Ctheta%3D%5Cfrac%7Be%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%7B1%2B%20e%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D
"\\theta=\\frac{e^{\\boldsymbol{\\beta X}}}{1+ e^{\\boldsymbol{\\beta X}}}").
The posterior distribution is ![P(\\theta|y)= L(\\theta)P(\\theta)
\\propto \\prod\_{i=1}^n \\Big(\\ \\frac{e^{\\boldsymbol{\\beta X}}}{1+
e^{\\boldsymbol{\\beta
X}}}\\Big)^{y\_i}\\Big(1-\\frac{e^{\\boldsymbol{\\beta X}}}{1+
e^{\\boldsymbol{\\beta
X}}}\\Big)^{1-y\_i}](https://ibm.codecogs.com/png.latex?P%28%5Ctheta%7Cy%29%3D%20L%28%5Ctheta%29P%28%5Ctheta%29%20%5Cpropto%20%5Cprod_%7Bi%3D1%7D%5En%20%5CBig%28%5C%20%5Cfrac%7Be%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%7B1%2B%20e%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%5CBig%29%5E%7By_i%7D%5CBig%281-%5Cfrac%7Be%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%7B1%2B%20e%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%5CBig%29%5E%7B1-y_i%7D
"P(\\theta|y)= L(\\theta)P(\\theta) \\propto \\prod_{i=1}^n \\Big(\\ \\frac{e^{\\boldsymbol{\\beta X}}}{1+ e^{\\boldsymbol{\\beta X}}}\\Big)^{y_i}\\Big(1-\\frac{e^{\\boldsymbol{\\beta X}}}{1+ e^{\\boldsymbol{\\beta X}}}\\Big)^{1-y_i}").

``` r
# create train
train <- observed
# one hot encode person_home_ownership
train$mortgage <- ifelse(as.character(train$person_home_ownership)=='MORTGAGE',1,0)
train$own <- ifelse(as.character(train$person_home_ownership)=='RENT',1,0)
train <- train %>% subset(select=-c(person_home_ownership))
# create predictor and response
train_X <- train %>% subset(select=-c(loan_status))
train_y <- train$loan_status
# scale train_X
norm_pars <- preProcess(train_X)
train_X <- predict(norm_pars,train_X)
```

``` r
# create test
test_logistic <- test
# one hot encode person_home_ownership
test_logistic$mortgage <- 
  ifelse(as.character(test_logistic$person_home_ownership)=='MORTGAGE',1,0)
test_logistic$own <- 
  ifelse(as.character(test_logistic$person_home_ownership)=='RENT',1,0)
test_logistic <- test_logistic %>% subset(select=-c(person_home_ownership))
# create predictor and response
test_logistic_X <- test_logistic %>% subset(select=-c(loan_status))
test_logistic_y <- test_logistic$loan_status
# scale train_X
test_logistic_X <- predict(norm_pars,test_logistic_X)
```

``` r
# function for making predictions
logistic_predictions <- function(X,betas){
  X_new <- X
  X_new$ones <- rep(1,nrow(X_new))
  sums <- colSums(t(X_new)*betas)
  predictions <- c()
  for (i in 1:length(sums)){
    predictions <- c(predictions, exp(sums[i])/(1+exp(sums[i])))
  }
  return(predictions)
}
# function for plotting predictions 
pred_plot <- function(pred,observe){
  ggplot() +
    geom_histogram(aes(x=prob,fill=observe,y=..density..),
                   bins=20,alpha=0.5,position='identity') +
    geom_density(aes(x=prob,fill=observe,color=observe),alpha=0.3) +
    scale_color_manual(name='Observed',
                       values=c('0'='deepskyblue2',
                                '1'='darkorange2'),
                       labels=c('No Default','Default')) +
    scale_fill_manual(name='Observed',
                      values=c('0'='deepskyblue2',
                               '1'='darkorange2'),
                      labels=c('No Default','Default')) +
    labs(title='Logistic Regression Probabilities') + 
    xlab('Probability') + ylab('Density') +
    theme(plot.title=element_text(face='bold'))
}
# function for plotting pr curve
pr_curve <- function(recall,precision){
  ggplot() +
    geom_line(aes(recall,precision),color='deepskyblue2') +
  labs(title='Precision-Recall Curve') + 
  xlab('Recall') + ylab('Precision') +
  theme(plot.title=element_text(face='bold'))
}
# function for finding cut off
find_cut_off <- function(precision, cutoff, recall){
  rate <- as.data.frame(cbind(Cutoff=cutoff,Precision=precision,Recall=recall))
  rate$distance <- sqrt((rate[,2]^2)+(rate[,3]^2))
  index <- which.min(rate$distance)
  return(rate$Cutoff[index])
}
```

## Log Posterior

The log of the posterior distribution is ![\\ell(\\theta|y)\\propto
\\sum\_{i=1}^nlog\\Big\[\\Big(\\ \\frac{e^{\\boldsymbol{\\beta X}}}{1+
e^{\\boldsymbol{\\beta
X}}}\\Big)^{y\_i}\\Big(1-\\frac{e^{\\boldsymbol{\\beta X}}}{1+
e^{\\boldsymbol{\\beta
X}}}\\Big)^{1-y\_i}\\Big\]](https://ibm.codecogs.com/png.latex?%5Cell%28%5Ctheta%7Cy%29%5Cpropto%20%5Csum_%7Bi%3D1%7D%5Enlog%5CBig%5B%5CBig%28%5C%20%5Cfrac%7Be%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%7B1%2B%20e%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%5CBig%29%5E%7By_i%7D%5CBig%281-%5Cfrac%7Be%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%7B1%2B%20e%5E%7B%5Cboldsymbol%7B%5Cbeta%20X%7D%7D%7D%5CBig%29%5E%7B1-y_i%7D%5CBig%5D
"\\ell(\\theta|y)\\propto \\sum_{i=1}^nlog\\Big[\\Big(\\ \\frac{e^{\\boldsymbol{\\beta X}}}{1+ e^{\\boldsymbol{\\beta X}}}\\Big)^{y_i}\\Big(1-\\frac{e^{\\boldsymbol{\\beta X}}}{1+ e^{\\boldsymbol{\\beta X}}}\\Big)^{1-y_i}\\Big]").

``` r
# function for calculating the log of the posterior
log_posterior <- function(beta){
  probs <- c()
  for (i in 1:nrow(train_X)){
    power <- beta[1]+(beta[2]*train_X[i,1])+(beta[3]*train_X[i,2])+
      (beta[4]*train_X[i,3])+(beta[5]*train_X[i,4])+(beta[6]*train_X[i,5])+
      (beta[7]*train_X[i,6])
    probs <- c(probs,exp(power)/(1+exp(power)))
  }
  if (any(probs==0)|any(probs==1)){
    return(-Inf)
  } else{
    total <- 0
    for (i in 1:length(probs)){
      total <- total + log((probs[i]^train_y[i])*((1-probs[i])^(1-train_y[i])))
    }
    return(total)
  }
}
```

## Metropolis Hastings Algorithm

Here are the steps for using the Metropolis Hastings Algorithm to find
![\\Beta](https://ibm.codecogs.com/png.latex?%5CBeta "\\Beta") values
for Bayesian Logistic Regression:

1.  Start a Markov Chain at
    ![\\theta\_0](https://ibm.codecogs.com/png.latex?%5Ctheta_0
    "\\theta_0")
2.  Generate
    ![\\theta^\*](https://ibm.codecogs.com/png.latex?%5Ctheta%5E%2A
    "\\theta^*") from the proposal distribution
    ![J(\\theta^\*|\\theta\_t)\\sim
    N(\\theta\_t,\\Sigma)](https://ibm.codecogs.com/png.latex?J%28%5Ctheta%5E%2A%7C%5Ctheta_t%29%5Csim%20N%28%5Ctheta_t%2C%5CSigma%29
    "J(\\theta^*|\\theta_t)\\sim N(\\theta_t,\\Sigma)")
3.  Compute
    ![r=min(0,log(P(\\theta^\*|y))-log(P(\\theta\_t|y)))](https://ibm.codecogs.com/png.latex?r%3Dmin%280%2Clog%28P%28%5Ctheta%5E%2A%7Cy%29%29-log%28P%28%5Ctheta_t%7Cy%29%29%29
    "r=min(0,log(P(\\theta^*|y))-log(P(\\theta_t|y)))")
4.  Generate a uniform random number
    ![u](https://ibm.codecogs.com/png.latex?u "u") from ![U\\sim
    Unif(0,1)](https://ibm.codecogs.com/png.latex?U%5Csim%20Unif%280%2C1%29
    "U\\sim Unif(0,1)")
5.  Set
    ![\\theta\_{t+1}\\leftarrow\\theta^\*](https://ibm.codecogs.com/png.latex?%5Ctheta_%7Bt%2B1%7D%5Cleftarrow%5Ctheta%5E%2A
    "\\theta_{t+1}\\leftarrow\\theta^*") with probability
    ![r](https://ibm.codecogs.com/png.latex?r "r")

<!-- end list -->

  - If
    ![log(u)\<log(r)](https://ibm.codecogs.com/png.latex?log%28u%29%3Clog%28r%29
    "log(u)\<log(r)"), accept
    ![\\theta^\*](https://ibm.codecogs.com/png.latex?%5Ctheta%5E%2A
    "\\theta^*") as the next sample, meaning
    ![\\theta\_{t+1}=\\theta\*](https://ibm.codecogs.com/png.latex?%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta%2A
    "\\theta_{t+1}=\\theta*")
      - Otherwise keep
        ![\\theta](https://ibm.codecogs.com/png.latex?%5Ctheta
        "\\theta") at its current value:
        ![\\theta\_{t+1}=\\theta\_t](https://ibm.codecogs.com/png.latex?%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta_t
        "\\theta_{t+1}=\\theta_t")

<!-- end list -->

``` r
# r logistic regression
logistic_regression <- function(beta_0,burnin,iter,sigma){
  betas <- NULL
  beta_t <- beta_0
  for (i in 1:iter){
    set.seed(143+i)
    beta_p <- mvtnorm::rmvnorm(1,mean=beta_t,sigma=sigma)
    log_r <- min(0,log_posterior(beta_p)-log_posterior(beta_t))
    u <- runif(1,min=0,max=1)
    if(log(u)<log_r){
      beta_t <- beta_p
    }
    betas <- rbind(betas,beta_t)
  }
  if (burnin!=0)
    betas <- betas %>% tail(-1*burnin)
  return(betas)
}
```

`{ function for finding best variance, very long to run} # function for
finding best variance find_var <- function(vars,beta_0){ results <- NULL
for (i in vars){ model <- logistic_regression(beta_0,750,2000,i*diag(7))
total <- 0 for (j in 1:7){ total <- total+effectiveSize(model[,j]) }
results <- rbind(results,c(i,total)) } return(results) } # find best
variance beta_0 <- rep(0,7) vars <-
find_var(c(0.005,0.006,0.007,0.009),beta_0)`

``` r
# train logistic regression model
beta_0 <- rep(0,7)
mh_logistic <- logistic_regression(beta_0,750,2000,0.005*diag(7))
# output effective sample size for each beta
for (i in 1:7){
  cat('Effective size for beta',i,' :',effectiveSize(mh_logistic[,i]),'\n')
}
```

    ## Effective size for beta 1  : 51.26528 
    ## Effective size for beta 2  : 22.61137 
    ## Effective size for beta 3  : 25.00869 
    ## Effective size for beta 4  : 46.99557 
    ## Effective size for beta 5  : 27.79148 
    ## Effective size for beta 6  : 5.045686 
    ## Effective size for beta 7  : 9.157431

The effective size for
![\\beta\_6](https://ibm.codecogs.com/png.latex?%5Cbeta_6 "\\beta_6")
and ![\\beta\_7](https://ibm.codecogs.com/png.latex?%5Cbeta_7
"\\beta_7") are fairly small.

``` r
# create trace plots
par(mfcol=c(3,3))
for (i in 1:7){
  coda::traceplot(as.mcmc(mh_logistic[,i]),col='deepskyblue2',
                  main=paste('Traceplot for Beta',i))
}
```

<img src="BayesianDataInference_files/figure-gfm/traceplot of each mh beta-1.png" style="display: block; margin: auto;" />

The traceplots for
![\\beta\_6](https://ibm.codecogs.com/png.latex?%5Cbeta_6 "\\beta_6")
and ![\\beta\_7](https://ibm.codecogs.com/png.latex?%5Cbeta_7
"\\beta_7") show high autocorrelation, meaning that a larger variance
for these ![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta
"\\beta")s would increase the number of samples.

``` r
# create density plots
par(mfcol=c(3,3))
for (i in 1:7){
  densplot(as.mcmc(mh_logistic[,i]),col='deepskyblue2',
       main=paste('Density Plot for Beta',i))
}
```

<img src="BayesianDataInference_files/figure-gfm/plot of each mh beta-1.png" style="display: block; margin: auto;" />

The density plot for each
![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta") is
roughly normal.

``` r
# output betas for logistic regression model
for (i in 1:7){
  cat('beta',i,' :',mean(mh_logistic[,i]),'\n')
}
```

    ## beta 1  : -1.614669 
    ## beta 2  : 0.2004199 
    ## beta 3  : -0.5971587 
    ## beta 4  : 1.473526 
    ## beta 5  : 1.494722 
    ## beta 6  : 1.482187 
    ## beta 7  : 1.467553

``` r
# create betas
mh_betas <- c()
for (i in 1:7) {
  mh_betas <- c(mh_betas,mean(mh_logistic[,i]))
}
# create data frame
prob <- logistic_predictions(train_X,mh_betas)
mh_pred_train <- data.frame(prob)
mh_pred_train$observe <- as.factor(train_y)
# plot predictions
pred_plot(mh_pred_train$prob,mh_pred_train$observe)
```

<img src="BayesianDataInference_files/figure-gfm/obtain mh predictions-1.png" style="display: block; margin: auto;" />

Observations with probabilities around 90% are likely to default.

``` r
# obtain precision and recall
mh_predict <- prediction(mh_pred_train$prob,mh_pred_train$observe)
mh_performance <- performance(mh_predict,'prec','rec')
# plot area under pr curve
pr_curve(performance(mh_predict, 'rec')@y.values[[1]],
         performance(mh_predict, 'prec')@y.values[[1]])
```

<img src="BayesianDataInference_files/figure-gfm/mh precision and recall curve-1.png" style="display: block; margin: auto;" />

``` r
# find cut off
mh_best <- find_cut_off(performance(mh_predict, 'prec')@y.values[[1]],
                        performance(mh_predict, 'prec')@x.values[[1]],
                        performance(mh_predict, 'rec')@y.values[[1]])
# output the result
cat('Cutoff: ',mh_best)
```

    ## Cutoff:  0.9332923

``` r
# use threshold to classify observations
mh_pred_train$pred <- ifelse(mh_pred_train$prob>=mh_best,'1','0')
# create confusion matrix
kable(table(pred=mh_pred_train$pred,true=mh_pred_train$observe),
      col.names=c('No Default','Default'))%>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

No Default

</th>

<th style="text-align:right;">

Default

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

1472

</td>

<td style="text-align:right;">

376

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

429

</td>

<td style="text-align:right;">

350

</td>

</tr>

</tbody>

</table>

``` r
# create data frame
prob <- logistic_predictions(test_logistic_X,mh_betas)
mh_pred_test <- data.frame(prob)
mh_pred_test$observe <- as.factor(test_logistic_y)
# make predictions
mh_pred_test$pred <- ifelse(mh_pred_test$prob>=mh_best,'1','0')
# create confusion matrix
kable(table(pred=mh_pred_test$pred,true=mh_pred_test$observe),
      col.names=c('No Default','Default'))%>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

No Default

</th>

<th style="text-align:right;">

Default

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

749

</td>

<td style="text-align:right;">

181

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

213

</td>

<td style="text-align:right;">

171

</td>

</tr>

</tbody>

</table>

The model has a 70.01% accuracy on the test set. It misclassifies
default observations more often than no default observations. This is
because the dataset is unbalanced and the model has not adequately
learned the trends in the defaulted observations.

## Hamiltonian Monte Carlo

HMC is a method for producing proposals that are accepted with high
probability by using Hamiltonian dynamics instead of a proposal
distribution. I also used this method to create a logistic regression
model.

``` r
# logistic regression stan model
logistic_model <- stan_model('logistic_regression.stan')
```

    ## Trying to compile a simple C file

    ## Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    ## clang -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -D_REENTRANT  -DBOOST_DISABLE_ASSERTS -DBOOST_PENDING_INTEGER_LOG2_HPP -include stan/math/prim/mat/fun/Eigen.hpp   -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -I/usr/local/include  -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:88:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:1: error: unknown type name 'namespace'
    ## namespace Eigen {
    ## ^
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:16: error: expected ';' after top level declarator
    ## namespace Eigen {
    ##                ^
    ##                ;
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    ## #include <complex>
    ##          ^~~~~~~~~
    ## 3 errors generated.
    ## make: *** [foo.o] Error 1

``` r
# fit posterior distribution
set.seed(136)
stan_fit <- rstan::sampling(
  logistic_model, data=
    list(N=nrow(train_X),y=train_y,sigma=10,K=6,X=train_X),refresh=0)
# samples from posterior distribution
logistic_model_samples <- rstan::extract(stan_fit)
```

``` r
# create trace plots
par(mfcol=c(3,2))
for (i in 1:6){
  coda::traceplot(as.mcmc(logistic_model_samples$beta[,i]),col='deepskyblue2',
                  main=paste('Traceplot for Beta',i))
}
```

<img src="BayesianDataInference_files/figure-gfm/traceplot of each hmc beta-1.png" style="display: block; margin: auto;" />

All traceplots show a low autocorrelation and a low rejection rate.

``` r
# create density plots
par(mfcol=c(3,2))
for (i in 1:6){
  densplot(as.mcmc(logistic_model_samples$beta[,i]),col='deepskyblue2',
       main=paste('Density Plot for Beta',i))
}
```

<img src="BayesianDataInference_files/figure-gfm/plot of each hmc beta-1.png" style="display: block; margin: auto;" />

All ![\\beta](https://ibm.codecogs.com/png.latex?%5Cbeta "\\beta")s have
observations following a roughly normal distribution.

``` r
# output betas for logistic regression model
cat('alpha:',mean(logistic_model_samples$alpha),'\n')
```

    ## alpha: -1.578634

``` r
for (i in 1:6) {
  cat('beta',i,': ',mean(logistic_model_samples$beta[,i]),'\n')
}
```

    ## beta 1 :  0.1272398 
    ## beta 2 :  -0.515685 
    ## beta 3 :  1.447984 
    ## beta 4 :  1.415756 
    ## beta 5 :  1.325408 
    ## beta 6 :  1.307114

``` r
hmc_betas <- c(mean(logistic_model_samples$alpha))
for (i in 1:6){
  hmc_betas <- c(hmc_betas,mean(logistic_model_samples$beta[,i]))
}
# create data frame
prob <- logistic_predictions(train_X,hmc_betas)
hmc_pred_train <- data.frame(prob)
hmc_pred_train$observe <- as.factor(train_y)
# plot predictions
pred_plot(hmc_pred_train$prob,hmc_pred_train$observe)
```

<img src="BayesianDataInference_files/figure-gfm/obtain hmc predictions-1.png" style="display: block; margin: auto;" />

Observations with probabilities greater than 90% often default.

``` r
# obtain precision and recall
hmc_predict <- prediction(hmc_pred_train$prob,hmc_pred_train$observe)
hmc_performance <- performance(hmc_predict,'prec','rec')
# plot area under pr curve
pr_curve(performance(hmc_predict, 'rec')@y.values[[1]],
         performance(hmc_predict, 'prec')@y.values[[1]])
```

<img src="BayesianDataInference_files/figure-gfm/hmc precision and recall curve-1.png" style="display: block; margin: auto;" />

``` r
# find cut off
hmc_best <- find_cut_off(performance(hmc_predict, 'prec')@y.values[[1]],
                         performance(hmc_predict, 'prec')@x.values[[1]],
                         performance(hmc_predict, 'rec')@y.values[[1]])
# output the result
cat('Cutoff: ',hmc_best)
```

    ## Cutoff:  0.9227504

``` r
# use threshold to classify observations
hmc_pred_train$pred <- ifelse(hmc_pred_train$prob>=hmc_best,'1','0')
# create confusion matrix
kable(table(pred=hmc_pred_train$pred,true=hmc_pred_train$observe),
      col.names=c('No Default','Default'))%>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

No Default

</th>

<th style="text-align:right;">

Default

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

1496

</td>

<td style="text-align:right;">

376

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

405

</td>

<td style="text-align:right;">

350

</td>

</tr>

</tbody>

</table>

``` r
# create data frame
prob <- logistic_predictions(test_logistic_X,hmc_betas)
hmc_pred_test <- data.frame(prob)
hmc_pred_test$observe <- as.factor(test_logistic_y)
# make predictions
hmc_pred_test$pred <- ifelse(hmc_pred_test$prob>=hmc_best,'1','0')
# create confusion matrix
kable(table(pred=hmc_pred_test$pred,true=hmc_pred_test$observe),
      col.names=c('No Default','Default'))%>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"), 
                full_width=FALSE,position='center')
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="text-align:left;">

</th>

<th style="text-align:right;">

No Default

</th>

<th style="text-align:right;">

Default

</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

0

</td>

<td style="text-align:right;">

765

</td>

<td style="text-align:right;">

182

</td>

</tr>

<tr>

<td style="text-align:left;">

1

</td>

<td style="text-align:right;">

197

</td>

<td style="text-align:right;">

170

</td>

</tr>

</tbody>

</table>

Similarly, since that dataset is imbalanced, the Hamiltonian Monte Carlo
method misclassifies default more often than it misclassifies no
default. The accuracy on the test set is 69.77%.

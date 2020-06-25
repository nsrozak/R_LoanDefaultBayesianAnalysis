
// logistic regression model
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
  real<lower=0> sigma;
  int K;
  matrix[N,K] X;
}
parameters {
  real alpha;
  vector[K] beta;
}
transformed parameters {
  vector[N] eta;
  eta = alpha + X*beta;
}
model {
  alpha ~ normal(0., sigma);
  beta ~ normal(0., sigma);
  y ~ bernoulli_logit(eta);
}

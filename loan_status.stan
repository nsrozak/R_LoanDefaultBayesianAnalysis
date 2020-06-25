
// loan status model
data {
  int<lower=0> J;  // data 
  int<lower=0> y[J];
  int<lower=0> n[J];
  real<lower=0> a;  // prior
  real<lower=0> b;
  real<lower=0> m;
  real<lower=0> std;
}
parameters {
  real<lower=0,upper=1> mu;
  real<lower=0> eta;
  real<lower=0,upper=1> theta[J];
}
transformed parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
  alpha = eta*mu;
  beta = eta*(1-mu);
}
model {
  mu ~ beta(a,b);
  eta ~ normal(m,std);
  theta ~ beta(alpha,beta);
  y ~ binomial(n,theta);
}

// the basic shrinkage model is from:
// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
// GB added imputation procedure and generated quantities block

data {
  int<lower=1> N;              // Number of data
  int<lower=1> M;              // Number of basic covariates
  matrix[N, M] X;
  real y[N];
  int M_quadratic;             // number of variables tfor scaling and quadratic terms
  real scale_sd[M+M_quadratic];  // stand deviations for scaling
  real scale_mu[M+M_quadratic];  // means for scaling
  int transf_idx[2];           // start and end idex of variables that need to be log-transformed
  int N_missing;               // number of missing values
  int miss_row[N_missing];     // row indices for missing values
  int miss_col[N_missing];     // column indices for missing values
  real m0;                     // Expected number of large slopes
  real slab_scale;             // Scale for large slopes

}

// slab_scale = 5, slab_df = 25 -> 8 divergences

transformed data {
  real slab_scale2 = square(slab_scale);
  real slab_df = 25;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
  // number raw predictors + number squared predictors 
  //(all scaled predictors are also entered squared)
  int MQ = M + M_quadratic;
  // number raw predictors + number squared predictors + interactions
  int MQI = MQ; // + N_interactions;
  matrix[N, MQI] X_scaled;

  X_scaled[,1:M] = X;
  // scale to mean 0, sd 1 (after log transform, if required)
  for (k in 1:M_quadratic) {
    if (k >= transf_idx[1]) {
      X_scaled[,k] = (log(X_scaled[,k])-scale_mu[k])/scale_sd[k];
    } else {
      X_scaled[,k] = (X_scaled[,k]-scale_mu[k])/scale_sd[k];
    }
    // add quadratic term
    for (n in 1:N) 
      X_scaled[n,M+k] = (X_scaled[n,k]^2-scale_mu[M+k])/scale_sd[M+k];
  }
}

parameters {
  vector[MQI] beta_tilde;
  vector<lower=0>[MQI] lambda;
  real<lower=0> c2_tilde;
  real<lower=0> tau_tilde;
  real alpha;
  real<lower=0> sigma;
  vector[N_missing] imputations;
}

transformed parameters {
  vector[MQI] beta;
  {
    real tau0 = (m0 / (MQI - m0)) * (sigma / sqrt(1.0 * N));
    real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)

    // c2 ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
    // Implies that marginally beta ~ student_t(slab_df, 0, slab_scale)
    real c2 = slab_scale2 * c2_tilde;

    vector[MQI] lambda_tilde =
      sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

    // beta ~ normal(0, tau * lambda_tilde)
    beta = tau * lambda_tilde .* beta_tilde;
  }
}

model {
  matrix[N, MQI] Xi = X_scaled;
  imputations ~ normal(0,2);
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 2);
  
  // impute missing values and add quadratic term
  // log-transformed data are imputed on the log scale
  // binary data are imputed to be between 0 and 1
  for (i in 1:N_missing) {
    Xi[miss_row[i],miss_col[i]] = miss_col[i] > M_quadratic ? 
                                  inv_logit(imputations[i]) : 
                                  imputations[i];
    if (miss_col[i] <= M_quadratic) {
       int i2 = M + miss_col[i];
       Xi[miss_row[i], i2] = (Xi[miss_row[i], miss_col[i]]^2-scale_mu[i2]) / 
                              scale_sd[i2];
    }
  }
  
  y ~ normal(Xi * beta + alpha, sigma);
}

generated quantities {
  matrix[N, MQI] Xi = X_scaled;
  real y_hat[N];
  // impute missing values and add quadratic term
  for (i in 1:N_missing) {
    Xi[miss_row[i],miss_col[i]] = miss_col[i] > M_quadratic ? 
                                  inv_logit(imputations[i]) : 
                                  imputations[i];
    if (miss_col[i] <= M_quadratic) {
       int i2 = M + miss_col[i];
       Xi[miss_row[i], i2] = (Xi[miss_row[i], miss_col[i]]^2-scale_mu[i2]) / 
                              scale_sd[i2];
    }
  }
  
  for (n in 1:N) y_hat[n] = normal_rng(dot_product(Xi[n,], beta) + alpha, sigma);
}


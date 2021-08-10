#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <Rmath.h>
#include <omp.h>

using namespace arma;
using namespace Rcpp;
using namespace std;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends("RcppArmadillo")]]


double U_theta(const arma::vec &theta_miss, const arma::vec &mu, const arma::mat &prec, const arma::vec &lambda0_miss, const arma::vec &lambda1_miss) {
  arma::vec U = (theta_miss - mu).t() * prec * (theta_miss - mu) / 2.0 -
    sum(log(normcdf(lambda0_miss + lambda1_miss % theta_miss)));
  return U(0);
}


void U_theta_grad(arma::vec &U_grad, const arma::vec &theta_miss, const arma::vec &mu, const arma::mat &prec, const arma::vec &lambda0_miss, const arma::vec &lambda1_miss) {
  U_grad = prec * (theta_miss - mu) - 
    (lambda1_miss % normpdf(lambda0_miss + lambda1_miss % theta_miss)) / normcdf(lambda0_miss + lambda1_miss % theta_miss);
}


void HMC_theta(arma::vec &theta_new, double &log_r, const arma::vec &theta_miss, const arma::vec &mu, const arma::mat &prec, const arma::vec &lambda0_miss,
               const arma::vec &lambda1_miss, const arma::vec &p_rnorm, const double &epsilon, const int &num_step) {
  //The kinetic energy has the simplest form sum(p^2)/2
  arma::vec U_grad;
  
  theta_new = theta_miss;
  arma::vec p_new = p_rnorm;
  
  // a half step for momentum at the beginning
  U_theta_grad(U_grad, theta_new, mu, prec, lambda0_miss, lambda1_miss);
  p_new -= epsilon *  U_grad / 2.0;
  
  // full steps for position and momentum
  for (int i = 0; i < (num_step-1); i++) {
    theta_new += epsilon * p_new;
    
    U_theta_grad(U_grad, theta_new, mu, prec, lambda0_miss, lambda1_miss);
    p_new -= epsilon * U_grad;
  }
  theta_new += epsilon * p_new;
  
  U_theta_grad(U_grad, theta_new, mu, prec, lambda0_miss, lambda1_miss);
  p_new -= epsilon * U_grad / 2.0;
  
  p_new = -p_new;
  log_r = U_theta(theta_miss, mu, prec, lambda0_miss, lambda1_miss) - U_theta(theta_new, mu, prec, lambda0_miss, lambda1_miss) +
    sum(p_rnorm % p_rnorm) / 2.0 - sum(p_new % p_new) / 2.0;
}


void update_theta(arma::mat &theta_t, const arma::mat &ind_zero, const arma::mat &mu_t, const arma::cube &invcov_t, const arma::cube &cov_t, const arma::vec &lambda0_t, 
                  const arma::vec &lambda1_t, const arma::vec &group_t, const int &N, const int &G, const double &epsilon, 
                  const int &num_step, const int &n_threads) {
  int i;
  
  arma::vec runif_vec = randu(N);
  arma::mat rnorm_mat = randn(G, N);
  
#pragma omp parallel shared(theta_t, runif_vec, rnorm_mat) private(i) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (i = 0; i < N; i++) {
    
    arma::vec ind_zero_i = ind_zero.row(i).t();
    uvec ind_0 = find(ind_zero_i == true);
    
    if (ind_0.n_elem > 0) {
      arma::vec theta_i = theta_t.row(i).t();
      uvec ind_obs = find(ind_zero_i == false);
      
      arma::vec theta_i_obs = theta_i(ind_obs);
      
      arma::vec theta_i_0 = theta_i(ind_0);
      
      arma::vec mu_i = mu_t.col(group_t(i));
      arma::mat cov_i = cov_t.slice(group_t(i));
      arma::mat invcov_i = invcov_t.slice(group_t(i));
      
      //calculate the acceptance probability
      arma::vec mu_obs = mu_i(ind_obs);
      arma::vec mu_0 = mu_i(ind_0);
      
      arma::mat invcov_i_21 = invcov_i.submat(ind_obs, ind_0);
      
      arma::mat prec_cond = invcov_i.submat(ind_0, ind_0);
      
      arma::mat cov_obs_inv = invcov_i.submat(ind_obs, ind_obs) - invcov_i_21 * inv(prec_cond) * invcov_i_21.t();
      
      arma::mat cov_21 = cov_i.submat(ind_0, ind_obs);
      arma::mat cov_0 = cov_i.submat(ind_0, ind_0);
      
      arma::vec mu_cond = mu_0 + cov_21 * cov_obs_inv * (theta_i_obs - mu_obs);
      
      arma::vec p_rnorm = rnorm_mat(span(0, (ind_0.n_elem - 1)), i);
      
      arma::vec lambda0_miss = lambda0_t(ind_0);
      arma::vec lambda1_miss = lambda1_t(ind_0);
      
      arma::vec theta_star_0;
      double log_r;
      HMC_theta(theta_star_0, log_r, theta_i_0, mu_cond, prec_cond, lambda0_miss, lambda1_miss, p_rnorm, epsilon, num_step);
      
      if (runif_vec(i) < exp(log_r)) {
        theta_i(ind_0) = theta_star_0;
        theta_t.row(i) = theta_i.t();
      }
    }
  }
}
}


void update_mu(arma::mat &mu_t, const arma::mat &theta_t, const arma::cube &invcov_t, const arma::vec &group_t, const int &G, const int &K, const double &eta_mu = 0, const double &tau_sq_mu = 1) {
  arma::mat I_tau_sq = eye(G,G);
  I_tau_sq /= tau_sq_mu;
  for (int k = 0; k < K; k++) {
    uvec ind_k = find(group_t == k);  
    arma::vec tmp1 = sum(theta_t.rows(ind_k), 0).t();
    
    arma::mat invcov_k = invcov_t.slice(k);
    
    arma::mat COV = inv(ind_k.n_elem * invcov_k + I_tau_sq);
    arma::vec MU = COV * (invcov_k * tmp1 + eta_mu/tau_sq_mu);
    
    arma::vec mu_k = arma::mvnrnd(MU, COV, 1);
    mu_t.col(k) = mu_k;
  }
}


void update_invcov(arma::cube &invcov_t, arma::cube &cov_t, const arma::cube &edge_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::vec &group_t, 
                   const double &ssp_v0, const double &ssp_v1, const double &ssp_l, const int &G, const int &K) {
  for (int k = 0; k < K; k++) {
    uvec ind_k = find(group_t == k);  
    arma::mat theta_k = theta_t.rows(ind_k).t();
    
    arma::mat theta_mu = theta_k.each_col() - mu_t.col(k);
    arma::mat S = theta_mu * theta_mu.t();
    
    arma::mat edge_k = edge_t.slice(k);
    arma::mat V = edge_k * ssp_v1 * ssp_v1;
    uvec ind_n_v1 = find(edge_k == 0);
    V(ind_n_v1).fill(ssp_v0 * ssp_v0); 
    V.diag().fill(0);
    
    arma::mat invcov_k = invcov_t.slice(k);
    arma::mat cov_k = cov_t.slice(k);
    
    arma::vec G_vec = regspace(0, G-1);
    
    for (int g = 0; g < G; g++) {
      uvec ind_ng = find(G_vec != g);
      arma::vec v12_g = V.col(g);
      arma::mat v12_inv = diagmat(1/v12_g(ind_ng));
      arma::vec cov_k_g = cov_k.col(g);
      arma::mat w11_inv = cov_k.submat(ind_ng, ind_ng) - cov_k_g(ind_ng) * cov_k_g(ind_ng).t() / cov_k_g(g);
      arma::mat C = (S(g,g) + ssp_l) * w11_inv + v12_inv;
      
      arma::mat C_chol_inv = inv(chol(C));
      
      arma::vec s12_g = S.col(g);
      arma::vec mu_w12 = -C_chol_inv * C_chol_inv.t() * s12_g(ind_ng);
      
      arma::vec rnorm_vec = randn(G-1);
      
      arma::vec w12 = mu_w12 + C_chol_inv * rnorm_vec;
      
      double w_v = randg(distr_param(ind_k.n_elem/2.0 + 1.0, 2.0/(S(g,g)+ssp_l)));
      arma::vec w11_inv_w12 = w11_inv * w12;
      arma::vec w22 = w_v + w12.t() * w11_inv_w12;
      
      arma::vec w_update(G);
      w_update(ind_ng) = w12;
      w_update(g) = w22(0);
      
      invcov_k.col(g) = w_update;
      invcov_k.row(g) = w_update.t();
      
      cov_k.submat(ind_ng, ind_ng) = w11_inv + w11_inv_w12 * w11_inv_w12.t() / w_v;
      arma::vec cov_k_ng = - w11_inv_w12 / w_v;
      
      arma::vec cov_update(G);
      cov_update(ind_ng) = cov_k_ng;
      cov_update(g) = 1 / w_v;
      
      cov_k.col(g) = cov_update;
      cov_k.row(g) = cov_update.t();
    }
    cov_t.slice(k) = cov_k;
    invcov_t.slice(k) = invcov_k;
  }
}


void update_edge(arma::cube &edge_t, const arma::cube &invcov_t, const double &ssp_v0, const double &ssp_v1, 
                 const double &ssp_pi, const int &G, const int &K, const int &n_threads) {
  int g;
  int t;
  arma::mat runif_mat = randu(K*(G-1), G);
#pragma omp parallel shared(edge_t, runif_mat) private(g, t) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {
    t = 0;
    for (int k = 0; k < K; k++) {
      for (int i = g+1; i < G; i++) {
        double log_p = invcov_t(g, i, k) * invcov_t(g, i, k) * (1.0 / (2.0*ssp_v1*ssp_v1) - 1.0 / (2.0*ssp_v0*ssp_v0)) +
          log(ssp_v1) - log(ssp_v0) + log(1.0 - ssp_pi) - log(ssp_pi);
        double r = 1.0/(1.0 + exp(log_p));
        double tmp = runif_mat(t, g);
        t += 1;
        if (tmp < r) {
          edge_t(g, i, k) = 1;
          edge_t(i, g, k) = 1;
        } else {
          edge_t(g, i, k) = 0;
          edge_t(i, g, k) = 0;
        }
      } 
    }
  }
}
}

double U_lam(const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, 
             const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1) {
  arma::vec U = - sum(log(1.0 - normcdf(lambda(0) + lambda(1) * theta_obs))) - sum(log(normcdf(lambda(0) + lambda(1) * theta_miss))) +
    (lambda(0) - lam0_0) * (lambda(0) - lam0_0) / (2.0 * sigma2_lam0) + (lambda(1) - lam1_0) * (lambda(1) - lam1_0) / (2.0 * sigma2_lam1);
  return U(0);
}


void U_lam_grad(arma::vec &U_grad, const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, 
                const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1) {
  arma::vec part1 = normpdf(lambda(0) + lambda(1) * theta_obs) / (1.0 - normcdf(lambda(0) + lambda(1) * theta_obs));
  arma::vec part2 = - normpdf(lambda(0) + lambda(1) * theta_miss) / normcdf(lambda(0) + lambda(1) * theta_miss);
  
  U_grad(0) = sum(part1) + sum(part2) + (lambda(0) - lam0_0) / sigma2_lam0;
  U_grad(1) = sum(part1 % theta_obs) + sum(part2 % theta_miss) + (lambda(1) - lam1_0) / sigma2_lam1;
}


void HMC_lam(arma::vec &lambda_new, double &log_r, const arma::vec &lambda, const arma::vec &theta_miss, const arma::vec &theta_obs, const arma::vec &p_rnorm,
             const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1,
             const double &epsilon, const int &num_step) {
  //The kinetic energy has the simplest form sum(p^2)/2
  arma::vec U_grad(2);
  
  lambda_new = lambda;
  arma::vec p_new = p_rnorm;
  
  // a half step for momentum at the beginning
  U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
  p_new -= epsilon *  U_grad / 2.0;
  // full steps for position and momentum
  for (int i = 0; i < (num_step-1); i++) {
    lambda_new += epsilon * p_new;
    
    U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
    p_new -= epsilon * U_grad;
  }
  lambda_new += epsilon * p_new;
  
  U_lam_grad(U_grad, lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1);
  p_new -= epsilon * U_grad / 2.0;
  
  p_new = -p_new;
  log_r = U_lam(lambda, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1) - 
    U_lam(lambda_new, theta_miss, theta_obs, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1) +
    sum(p_rnorm % p_rnorm) / 2.0 - sum(p_new % p_new) / 2.0;
  
}



void update_lambda(arma::vec &lambda0_t, arma::vec &lambda1_t, const arma::mat &theta_t, const arma::mat &ind_zero, const int &G,
                   const double &lam0_0, const double &lam1_0, const double &sigma2_lam0, const double &sigma2_lam1,
                   const double &epsilon, const int &num_step, const int &n_threads) {
  int g;
  arma::mat rnorm_mat = randn(2, G);
  arma::vec runif_vec = randu(G);
#pragma omp parallel shared(lambda0_t, lambda1_t, rnorm_mat, runif_vec) private(g) num_threads(n_threads)
{
#pragma omp for schedule(auto)
  for (g = 0; g < G; g++) {
    arma::vec lambda_g(2);
    lambda_g(0) = lambda0_t(g);
    lambda_g(1) = lambda1_t(g);
    
    arma::vec theta_g = theta_t.col(g);
    uvec ind_0= find(ind_zero.col(g) == true);
    uvec ind_obs = find(ind_zero.col(g) == false);
    
    arma::vec theta_g_obs = theta_g(ind_obs);
    
    arma::vec theta_g_miss = theta_g(ind_0);
    
    arma::vec p_rnorm = rnorm_mat.col(g);
    
    double tmp_unif = runif_vec(g);
    
    arma::vec lambda_g_new;
    double log_r;
    HMC_lam(lambda_g_new, log_r, lambda_g, theta_g_miss, theta_g_obs, p_rnorm, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon, num_step);
    
    if (tmp_unif < exp(log_r) && lambda_g_new(1) < 0) {
      lambda0_t(g) = lambda_g_new(0);
      lambda1_t(g) = lambda_g_new(1);
    }
  }
}
}

arma::vec rDirichlet(const arma::vec &alpha_vec) {
  arma::vec tmp(alpha_vec.n_elem, fill::zeros);
  for (unsigned int i = 0; i < alpha_vec.n_elem; i++) {
    tmp(i) = log(randg(distr_param(alpha_vec(i), 1.0)));
  }
  tmp = tmp - max(tmp);
  tmp = exp(tmp);
  tmp = tmp/sum(tmp);
  return tmp;
}


void update_pi(arma::vec &pi_t, const arma::vec &group_t, const arma::vec &gam, const int &K) {
  arma::vec tmp0(K, fill::zeros);
  for (int k = 0; k < K; k++) {
    uvec ind_j = find(group_t == k);
    tmp0(k) = ind_j.n_elem;
  }
  pi_t = rDirichlet(tmp0+gam);
}


void update_group(arma::vec &group_t, const arma::cube &invcov_t, const arma::mat &theta_t, const arma::mat &mu_t, const arma::vec &pi_t, 
                  const int &G, const int &N, const int &K, const int &n_threads) {
  uvec pi_n0 = find(pi_t > 0);
  int i;
  arma::vec invcov_logdet(K);
  for (int k = 0; k < K; k++) {
    invcov_logdet(k) = log(det(invcov_t.slice(k)))/2.0;
  }
  
  arma::mat prob_Cell_mat(K, N);
  
#pragma omp parallel shared(pi_n0, invcov_logdet, prob_Cell_mat) private(i) num_threads(n_threads)
{
#pragma omp for schedule(auto) 
  for (i = 0; i < N; i++) {
    arma::vec theta_i = theta_t.row(i).t();
    arma::vec tmp(K);
    tmp.fill(- datum::inf);
    for (unsigned int k_pi = 0; k_pi < pi_n0.n_elem; k_pi++) {
      arma::vec tmp_k_pi = invcov_logdet(pi_n0(k_pi)) - 
        (theta_i - mu_t.col(pi_n0(k_pi))).t() * invcov_t.slice(pi_n0(k_pi)) * (theta_i - mu_t.col(pi_n0(k_pi)))/2.0;
      tmp(pi_n0(k_pi)) = tmp_k_pi(0);
    }
    tmp.replace(datum::inf, pow(10, 308));
    arma::vec tmp_new = tmp - max(tmp);
    tmp_new = exp(tmp_new);
    arma::vec prob = tmp_new % pi_t;
    prob = prob / sum(prob);
    
    prob_Cell_mat.col(i) = prob;
  }
}

arma::vec Cell_range = regspace(0, 1, K-1);
for (i = 0; i < N; i++) {
  arma::vec prob_cell = prob_Cell_mat.col(i);
  group_t(i) = Rcpp::RcppArmadillo::sample(Cell_range, 1, false, prob_cell)(0);
}
}


// [[Rcpp::export]]
List MCMC_full(const int num_iter, const int num_save, arma::mat theta_t, arma::mat ind_zero, arma::mat mu_t,
               arma::cube invcov_t, arma::cube cov_t, arma::cube edge_t, arma::vec group_t, arma::vec lambda0_t, arma::vec lambda1_t, 
               arma::vec pi_t, arma::vec gam, const int G, const int N, const int K, 
               double ssp_v0, const double ssp_v1, const double ssp_l, double ssp_pi,
               double epsilon_theta = 0.2, int num_step_theta = 20,
               double eta_mu = 0, double tau_sq_mu = 1,
               double lam0_0=2, double lam1_0=-2, double sigma2_lam0=0.25, 
               double sigma2_lam1=0.25, double epsilon_lam = 0.01, int num_step_lam = 10,
               bool iter_save = false, int n_threads = 1, int iter_print = 1000, bool class_print = false) {
  
  group_t = group_t - 1;
  
  int save_start = num_iter - num_save;
  arma::mat group_save(K, N, fill::zeros);
  arma::mat mu_save(G, K, fill::zeros);
  arma::cube invcov_save(G, G, K, fill::zeros);
  arma::cube edge_save(G, G, K, fill::zeros);
  
  arma::vec Pi_save(K, fill::zeros);
  arma::mat theta_save(N, G, fill::zeros);
  arma::vec lam0_save(G, fill::zeros);
  arma::vec lam1_save(G, fill::zeros);
  
  if (iter_save) {
    arma::field<cube> invcov_iter(num_save);
    arma::field<cube> edge_iter(num_save);
    arma::cube theta_iter(N, G, num_save);

    arma::mat group_iter(N, num_save);
    arma::cube mu_iter(G, K, num_save);
    arma::mat lam0_iter(G, num_save);
    arma::mat lam1_iter(G, num_save);
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP Cell_table = table(Rcpp::_["..."] = group_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl; 
      
      for (int t_iter = 0; t_iter < num_iter; t_iter++) {
        update_theta(theta_t, ind_zero, mu_t, invcov_t, cov_t, lambda0_t, lambda1_t, group_t, N, G, epsilon_theta, num_step_theta, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam, n_threads);
        
        update_mu(mu_t, theta_t, invcov_t, group_t, G, K, eta_mu, tau_sq_mu);
        
        update_invcov(invcov_t, cov_t, edge_t, theta_t, mu_t, group_t, ssp_v0, ssp_v1, ssp_l, G, K);
        
        update_edge(edge_t, invcov_t, ssp_v0, ssp_v1, ssp_pi, G, K, n_threads);
        
        update_group(group_t, invcov_t, theta_t, mu_t, pi_t, G, N, K, n_threads);
        
        update_pi(pi_t, group_t, gam, K);
        
        SEXP Cell_table = table(Rcpp::_["..."] = group_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int C_i = 0; C_i < N; C_i++) {
            group_save(group_t(C_i),C_i) += 1;
          }
          
          mu_save += mu_t;
          invcov_save += invcov_t;
          edge_save += edge_t;
          
          Pi_save += pi_t;
          theta_save += theta_t;
          
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          group_iter.col(save_i) = group_t;
          mu_iter.slice(save_i) = mu_t;
          
          invcov_iter(save_i) = invcov_t;
          edge_iter(save_i) = edge_t;
          
          theta_iter.slice(save_i) = theta_t;
          
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
        }
      }
      
      mu_save /= num_save;
      invcov_save /= num_save;
      edge_save /= num_save;
      
      Pi_save /= num_save;
      theta_save /= num_save;
      
      lam0_save /= num_save;
      lam1_save /= num_save;
      
      arma::vec group_est(N);
      for (int C_i = 0; C_i < N; C_i++) {
        group_est(C_i) = group_save.col(C_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=group_est, Named("cell_type_mean_expr")=mu_save, Named("cell_type_precision_matr")=invcov_save, Named("cell_type_edge_post_prob")=edge_save,
                                Named("prop")=Pi_save, Named("theta")=theta_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save,
                                Named("cell_type_mean_expr_post")=mu_iter, Named("cell_type_precision_matr_post")=invcov_iter, Named("lam0_post")=lam0_iter, Named("lam1_post")=lam1_iter,
                                Named("cell_label_post")=group_iter, Named("cell_type_edge_indicator_post")=edge_iter, Named("theta_post")=theta_iter);
    } else {
      
      for (int t_iter = 0; t_iter < num_iter; t_iter++) {
        update_theta(theta_t, ind_zero, mu_t, invcov_t, cov_t, lambda0_t, lambda1_t, group_t, N, G, epsilon_theta, num_step_theta, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam, n_threads);
        
        update_mu(mu_t, theta_t, invcov_t, group_t, G, K, eta_mu, tau_sq_mu);
        
        update_invcov(invcov_t, cov_t, edge_t, theta_t, mu_t, group_t, ssp_v0, ssp_v1, ssp_l, G, K);
        
        update_edge(edge_t, invcov_t, ssp_v0, ssp_v1, ssp_pi, G, K, n_threads);
        
        update_group(group_t, invcov_t, theta_t, mu_t, pi_t, G, N, K, n_threads);
        
        update_pi(pi_t, group_t, gam, K);
        
        if ((t_iter+1) % iter_print == 0) {
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          int save_i = t_iter - save_start;
          for (int C_i = 0; C_i < N; C_i++) {
            group_save(group_t(C_i),C_i) += 1;
          }
          
          mu_save += mu_t;
          invcov_save += invcov_t;
          edge_save += edge_t;
          
          Pi_save += pi_t;
          theta_save += theta_t;
          
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
          
          group_iter.col(save_i) = group_t;
          mu_iter.slice(save_i) = mu_t;
          
          invcov_iter(save_i) = invcov_t;
          edge_iter(save_i) = edge_t;
          
          theta_iter.slice(save_i) = theta_t;
          
          lam0_iter.col(save_i) = lambda0_t;
          lam1_iter.col(save_i) = lambda1_t;
        }
      }
      
      mu_save /= num_save;
      invcov_save /= num_save;
      edge_save /= num_save;
      
      Pi_save /= num_save;
      theta_save /= num_save;
      
      lam0_save /= num_save;
      lam1_save /= num_save;
      
      arma::vec group_est(N);
      for (int C_i = 0; C_i < N; C_i++) {
        group_est(C_i) = group_save.col(C_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=group_est, Named("cell_type_mean_expr")=mu_save, Named("cell_type_precision_matr")=invcov_save, Named("cell_type_edge_post_prob")=edge_save,
                                Named("prop")=Pi_save, Named("theta")=theta_save, Named("lam0")=lam0_save, Named("lam1")=lam1_save,
                                      Named("cell_type_mean_expr_post")=mu_iter, Named("cell_type_precision_matr_post")=invcov_iter, Named("lam0_post")=lam0_iter, Named("lam1_post")=lam1_iter,
                                      Named("cell_label_post")=group_iter, Named("cell_type_edge_indicator_post")=edge_iter, Named("theta_post")=theta_iter);
    }
  } else {
    
    if (class_print) {
      Environment base("package:base"); 
      Function table = base["table"];
      
      SEXP Cell_table = table(Rcpp::_["..."] = group_t);
      irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
      cout<<"Iteration "<< 0 <<endl;
      cout<<"Cell"<<endl; 
      cout<<Cell_out<<endl; 
      
      for (int t_iter = 0; t_iter < num_iter; t_iter++) {
        update_theta(theta_t, ind_zero, mu_t, invcov_t, cov_t, lambda0_t, lambda1_t, group_t, N, G, epsilon_theta, num_step_theta, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam, n_threads);
        
        update_mu(mu_t, theta_t, invcov_t, group_t, G, K, eta_mu, tau_sq_mu);
        
        update_invcov(invcov_t, cov_t, edge_t, theta_t, mu_t, group_t, ssp_v0, ssp_v1, ssp_l, G, K);
        
        update_edge(edge_t, invcov_t, ssp_v0, ssp_v1, ssp_pi, G, K, n_threads);
        
        update_group(group_t, invcov_t, theta_t, mu_t, pi_t, G, N, K, n_threads);
        
        update_pi(pi_t, group_t, gam, K);
        
        SEXP Cell_table = table(Rcpp::_["..."] = group_t);
        irowvec Cell_out = Rcpp::as<arma::irowvec>(Cell_table);
        cout<<"Iteration "<< t_iter+1 <<endl;
        cout<<"Cell"<<endl; 
        cout<<Cell_out<<endl; 
        
        if (t_iter >= save_start) {
          for (int C_i = 0; C_i < N; C_i++) {
            group_save(group_t(C_i),C_i) += 1;
          }
          
          mu_save += mu_t;
          invcov_save += invcov_t;
          edge_save += edge_t;
          
          Pi_save += pi_t;
          theta_save += theta_t;
          
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
        }
      }
      
      mu_save /= num_save;
      invcov_save /= num_save;
      edge_save /= num_save;
      
      Pi_save /= num_save;
      theta_save /= num_save;
      
      lam0_save /= num_save;
      lam1_save /= num_save;
      
      arma::vec group_est(N);
      for (int C_i = 0; C_i < N; C_i++) {
        group_est(C_i) = group_save.col(C_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=group_est, Named("cell_type_mean_expr")=mu_save, Named("cell_type_precision_matr")=invcov_save,
                                Named("cell_type_edge_post_prob")=edge_save, Named("prop")=Pi_save, Named("theta")=theta_save,
                                Named("lam0")=lam0_save, Named("lam1")=lam1_save);
    } else {
      
      for (int t_iter = 0; t_iter < num_iter; t_iter++) {
        update_theta(theta_t, ind_zero, mu_t, invcov_t, cov_t, lambda0_t, lambda1_t, group_t, N, G, epsilon_theta, num_step_theta, n_threads);
        
        update_lambda(lambda0_t, lambda1_t, theta_t, ind_zero, G, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam, n_threads);
        
        update_mu(mu_t, theta_t, invcov_t, group_t, G, K, eta_mu, tau_sq_mu);
        
        update_invcov(invcov_t, cov_t, edge_t, theta_t, mu_t, group_t, ssp_v0, ssp_v1, ssp_l, G, K);
        
        update_edge(edge_t, invcov_t, ssp_v0, ssp_v1, ssp_pi, G, K, n_threads);
        
        update_group(group_t, invcov_t, theta_t, mu_t, pi_t, G, N, K, n_threads);
        
        update_pi(pi_t, group_t, gam, K);
        
        if ((t_iter+1) % iter_print == 0) {
          cout<<"Iteration "<< t_iter+1 <<endl;
        }
        
        if (t_iter >= save_start) {
          for (int C_i = 0; C_i < N; C_i++) {
            group_save(group_t(C_i),C_i) += 1;
          }
          
          mu_save += mu_t;
          invcov_save += invcov_t;
          edge_save += edge_t;
          
          Pi_save += pi_t;
          theta_save += theta_t;
          
          lam0_save += lambda0_t;
          lam1_save += lambda1_t;
        }
      }
      
      mu_save /= num_save;
      invcov_save /= num_save;
      edge_save /= num_save;
      
      Pi_save /= num_save;
      theta_save /= num_save;
      
      lam0_save /= num_save;
      lam1_save /= num_save;
      
      arma::vec group_est(N);
      for (int C_i = 0; C_i < N; C_i++) {
        group_est(C_i) = group_save.col(C_i).index_max() + 1;
      }
      
      return List::create(Named("cell_labels")=group_est, Named("cell_type_mean_expr")=mu_save, Named("cell_type_precision_matr")=invcov_save,
                                Named("cell_type_edge_post_prob")=edge_save, Named("prop")=Pi_save, Named("theta")=theta_save, 
                                Named("lam0")=lam0_save, Named("lam1")=lam1_save);
    }
    
  }
}


// [[Rcpp::export]]
arma::vec update_pi_R(arma::vec group_t, arma::vec gam, int K) {
  arma::vec pi_new(K);
  arma::vec tmp0(K, fill::zeros);
  for (int k = 0; k < K; k++) {
    uvec ind_j = find(group_t == k);
    tmp0(k) = ind_j.n_elem;
  }
  pi_new = rDirichlet(tmp0+gam);
  return pi_new;
}

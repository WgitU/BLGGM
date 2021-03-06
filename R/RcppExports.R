# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

MCMC_full <- function(num_iter, num_save, theta_t, ind_zero, mu_t, invcov_t, cov_t, edge_t, group_t, lambda0_t, lambda1_t, pi_t, gam, G, N, K, ssp_v0, ssp_v1, ssp_l, ssp_pi, epsilon_theta = 0.2, num_step_theta = 20L, eta_mu = 0, tau_sq_mu = 1, lam0_0 = 2, lam1_0 = -2, sigma2_lam0 = 0.25, sigma2_lam1 = 0.25, epsilon_lam = 0.01, num_step_lam = 10L, iter_save = FALSE, n_threads = 1L, iter_print = 1000L, class_print = FALSE) {
    .Call(`_BLGGM_MCMC_full`, num_iter, num_save, theta_t, ind_zero, mu_t, invcov_t, cov_t, edge_t, group_t, lambda0_t, lambda1_t, pi_t, gam, G, N, K, ssp_v0, ssp_v1, ssp_l, ssp_pi, epsilon_theta, num_step_theta, eta_mu, tau_sq_mu, lam0_0, lam1_0, sigma2_lam0, sigma2_lam1, epsilon_lam, num_step_lam, iter_save, n_threads, iter_print, class_print)
}

update_pi_R <- function(group_t, gam, K) {
    .Call(`_BLGGM_update_pi_R`, group_t, gam, K)
}


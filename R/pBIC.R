#' Modified Baysian information criterion
#'
#' The function pBIC calculates the value of a modified Baysian information criterion considering model sparsity, which can be used to find the optimal cell type number.
#' 
#' @param BLGGM_result The result of function "BLGGM".
#' @param scRNA_data_matr The scRNA-seq data matrix, where rows represent genes and columns represent cells. Matrix values need to be normlaized single-cell expression data.
#'
#' @return pBIC returns the value of a modified Baysian information criterion.
#' @examples 
#' library(BLGGM)
#' 
#' #import example data
#' data(example_data)
#' 
#' #gene number
#' nrow(scRNA_data_matr)
#' 
#' #cell number
#' ncol(scRNA_data_matr)
#' 
#' #cell type number
#' n_celltype <- dim(precision_matr)[3]
#' 
#' #run BLGGM
#' set.seed(20201116)
#' Result <- BLGGM(scRNA_data_matr, n_celltype, num_iterations = 10000, num_threads = 10)
#' 
#' #pBIC
#' pBIC(Result, scRNA_data_matr)
#' 
#' @export
pBIC <- function(BLGGM_result, scRNA_data_matr) {
  N <- length(BLGGM_result$cell_labels)
  G <- dim(BLGGM_result$cell_type_mean_expr)[1]
  K <- dim(BLGGM_result$cell_type_mean_expr)[2]
  ind_zero <-t(scRNA_data_matr == 0)
  theta_est <- t(BLGGM_result$theta)
  
  tmp <- 0
  for (i in 1:N) {
    mis_i <- ind_zero[i, ]
    tmp_i <- 0
    if (sum(mis_i) > 0) {
      for (k in 1:K) {
        mu_mis <- BLGGM_result$cell_type_mean_expr[mis_i,k]
        mu_obs <- BLGGM_result$cell_type_mean_expr[!mis_i,k]
        invcov_mis <- matrix(BLGGM_result$cell_type_precision_matr[mis_i,mis_i,k],
                             sum(mis_i), sum(mis_i))
        invcov_obs <- matrix(BLGGM_result$cell_type_precision_matr[!mis_i,!mis_i,k], 
                             sum(!mis_i), sum(!mis_i))
        invcov_12 <- matrix(BLGGM_result$cell_type_precision_matr[!mis_i,mis_i,k],
                            sum(!mis_i), sum(mis_i))
        theta_mu_obs <- theta_est[i, !mis_i] - mu_obs
        
        inv_invcov_mis <- solve(invcov_mis)
        invcov_12_i22 <- invcov_12 %*% inv_invcov_mis
        inv_cov_obs <- invcov_obs - invcov_12_i22 %*% t(invcov_12)
        cov_12 <- -solve(inv_cov_obs) %*% invcov_12_i22
        cov_mis <- inv_invcov_mis - t(invcov_12_i22) %*% cov_12
        
        log_obs <- sum(log(1 - pnorm(BLGGM_result$lam0[!mis_i] + BLGGM_result$lam1[!mis_i] * theta_est[i, !mis_i]))) - 
          t(theta_mu_obs) %*% inv_cov_obs %*% theta_mu_obs / 2.0 - sum(!mis_i) * log(2*pi) / 2.0 + log(det(inv_cov_obs)) / 2.0
        
        mu_k_star <- mu_mis + t(cov_12) %*% inv_cov_obs %*% theta_mu_obs
        cov_k_star <- cov_mis - t(cov_12) %*% inv_cov_obs %*% cov_12
        
        theta_mis_MC <- t(mvrnorm(500, mu_k_star, cov_k_star))
        mis_MC <- mean(exp(colSums(pnorm(BLGGM_result$lam0[mis_i] + BLGGM_result$lam1[mis_i] * theta_mis_MC, log.p = T))))
        
        tmp_i <- tmp_i + BLGGM_result$prop[k] * exp(log_obs) * mis_MC
      }
    } else {
      for (k in 1:K) {
        mu_obs <- BLGGM_result$cell_type_mean_expr[,k]
        invcov_obs <- BLGGM_result$cell_type_precision_matr[,,k]
        theta_mu_obs <- theta_est[i, ] - mu_obs
        
        log_obs <- sum(log(1 - pnorm(BLGGM_result$lam0[!mis_i] + BLGGM_result$lam1[!mis_i] * theta_est[i, !mis_i]))) - 
          t(theta_mu_obs) %*% invcov_obs %*% theta_mu_obs / 2.0 - sum(!mis_i) * log(2*pi) / 2.0 + log(det(invcov_obs)) / 2.0
        
        tmp_i <- tmp_i + BLGGM_result$prop[k] * exp(log_obs)
      }
    }
    if (tmp_i == 0) {
      tmp <- tmp - 10000
    } else {
      tmp <- tmp + log(tmp_i)
    }
  }
  
  Zero_num <- 0
  for (k in 1:K) {
    Zero_num <- Zero_num  + sum(BLGGM_result$cell_type_adj_matr[,,k][upper.tri(BLGGM_result$cell_type_adj_matr[,,k])] == 0)
  }
  
  BIC_result <- -2 * tmp + (K - 1 + G * (2 + K + (G + 1) * K / 2.0) - Zero_num) * log(N)
  return(BIC_result)
}
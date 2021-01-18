#' Estimating Heterogeneous Gene Regulatory Networks from Zero-Inflated Single-Cell Expression Data
#'
#' The function BLGGM implements a Bayesian latent mixture GGM method to jointly estimate multiple gene regulatory networks accounting for the zero-inflation and unknown heterogeneity of single-cell expression data. It employs a hybrid Markov chain Monte Carlo algorithm to perform posterior inference.
#' 
#' @param scRNA_data_matr The scRNA-seq data matrix, where rows represent genes and columns represent cells. Matrix values need to be normalized single expression data.
#' @param n_celltype An integer, denoting the number of cell types.
#' @param warm_cluster_label_init The initialization of cluster labels is random or based on k-means. The default is FALSE, corresponding to the random initialization.
#' @param num_iterations The number of Gibbs sampler iterations. The default is 10000.
#' @param num_burnin The number of iterations in burn-in, after which the posterior samples are used to estimate the unknown parameters. The default is the first half of total iterations.
#' @param collect_post_sample Logical, collect the posterior samples or not. If users are only interested in the estimates, set it as FALSE to save the memory. If users would like to use posterior samples to construct credible intervals or for other uses, set it as TRUE. The default is FALSE.
#' @param hyperparameters A vector, which indicates 14 hyper-parameters in the priors or proposal distributions. The first two elements correspond to the step size and step number of leapfrog iteration for \eqn{\theta} respectively. The third and fourth elements are the mean and variance of the normal prior distribution for \eqn{\mu} respectively. The fifth and sixth elements are, respectively, the means of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1}. The seventh and eighth elements correspond to the variances of the normal priors for \eqn{\lambda_0} and \eqn{\lambda_1} respectively. The ninth and tenth elements are the step size and step number of leapfrog iteration for \eqn{\lambda} respectively. The eleventh and twelfth elements represent the variances of normal distributions in the spike-slab prior respectively. The thirteenth element is twice rate of exponential distributions in the spike-slab prior. The last one is the edge inclusion probability in the spike-slab prior. The default is 0.2, 20, 0, 1, 3, -2, 0.2, 0.2, 0.01, 10, 0.02, 1, 1, 2/(dim(scRNA_data_matr)[1] - 1)
#' @param hyperparameters_conc A vector is a vector of n_celltype dimensions, describing the parameters of Dirichlet prior for \eqn{\pi}. The default is a vector filled with one.
#' @param inclusion_thr A numeric between 0 and 1, denoting the threshold of edge inclusion. The edge with posterior probability of inclusion above the threshold is selected. 
#' @param print_label Logical, whether or not to print summarized cell type labels after each iteration. The default is FALSE.
#' @param print_per_iteration An integer (denote it by v), print the iteration information per v iterations when print_label is FALSE. The default is 1000.
#' @param num_threads An integer, denoting the number of threads used in the parallel computing. The default is 1.
#' 
#' @return BLGGM returns an R list including the following information.
#' \item{cell_labels}{A vector, indicating the estimated cell type labels for each cell.}
#' \item{cell_type_mean_expr}{A matrix of cell type expression profiles, in which rows are genes and columns correspond to cell types.}
#' \item{cell_type_precision_matr}{An array of cell-type-specific precision matrices, in which the third index corresponds to cell types.}
#' \item{cell_type_adj_matr}{An array of cell-type-specific gene adjacency matrix, in which the third index corresponds to cell types. 1 represents an edge, and 0 indicates no edge.}
#' \item{cell_type_edge_post_prob}{An array of cell-type-specific posterior probability matrices of edge inclusion, in which the third index corresponds to cell types.}
#' \item{prop}{A vector of estimated cell type proportions.}
#' \item{theta}{A matrix of \eqn{\theta}, in which rows are genes and columns correspond to cells.}
#' \item{lam0}{A vector, the estimated \eqn{\lambda_0} for each gene.}
#' \item{lam1}{A vector, the estimated \eqn{\lambda_1} for each gene.}
#' \item{cell_label_post}{Collected posterior samples of cell_labels when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{cell_type_mean_expr_post}{Collected sampls of cell_type_mean_expr when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam0_post}{Collected posterior samples of lam0 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{lam1_post}{Collected posterior samples of lam1 when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{theta_post}{Collected posterior samples of \eqn{\theta} when collect_post_sample is TRUE. Rows are cells and columns are genes. If collect_post_sample is FALSE, this output does not exist.}
#' \item{cell_type_precision_matr_post}{Collected posterior samples of cell_type_precision_matr when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' \item{cell_type_edge_indicator_post}{Collected posterior samples of edge inclusion indicators for each cell type when collect_post_sample is TRUE. If collect_post_sample is FALSE, this output does not exist.}
#' 
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
#' t1 <- Sys.time()
#' Result <- BLGGM(scRNA_data_matr, n_celltype, num_iterations = 10000, num_threads = 10)
#' t2 <- Sys.time()
#' 
#' #time cost
#' print(t2 - t1)
#' 
#' #Compared with true cell type labels
#' cell_table <- table(Result$cell_labels, cell_label_truth)
#' cell_table
#' 
#' #The following shows the Frobenius norms of precision matrix within each cell type
#' sqrt(sum((Result$cell_type_precision_matr[,,which.max(cell_table[,1])] - precision_matr[,,1])^2))
#' sqrt(sum((Result$cell_type_precision_matr[,,which.max(cell_table[,2])] - precision_matr[,,2])^2))
#' 
#' @references Qiuyu Wu, and Xiangyu Luo. "Estimating Heterogeneous Gene Regulatory Networks from Zero-Inflated Single-Cell Expression Data."
#' @export
BLGGM <- function(scRNA_data_matr, n_celltype, warm_cluster_label_init = FALSE, 
                  num_iterations = 10000, num_burnin = floor(num_iterations/2), collect_post_sample = FALSE, 
                  hyperparameters = c(0.2, 20, 0, 1, 3, -2, 0.2, 0.2, 0.01, 10, 0.02, 1, 1, 2/(dim(scRNA_data_matr)[1] - 1)),
                  hyperparameters_conc = rep(1, n_celltype), inclusion_thr = 0.5,
                  print_label = FALSE, print_per_iteration = 1000, num_threads = 1) {
  
  scRNA_data_matr <- as.matrix(scRNA_data_matr)
  
  #zero index
  ind_zero <- (scRNA_data_matr == 0)
  
  if ((sum(rowMeans(ind_zero) == 1)) > 0) {
    print("Warning: remove the genes having zero expression across spots or cells")
    ind_gene <- (rowMeans(ind_zero) < 1)
    scRNA_data_matr <- scRNA_data_matr[ind_gene, ]
    ind_zero <- ind_zero[ind_gene, ]
  }
  
  G <- dim(scRNA_data_matr)[1]
  n <- dim(scRNA_data_matr)[2]
  
  #initialize theta_t
  theta_t <- scRNA_data_matr
  for (g in 1:G) {
    theta_t[g, ind_zero[g, ]] <- quantile(scRNA_data_matr[g,!ind_zero[g,]], probs = 0.05)
  }
  theta_t <- log(theta_t)
  
  #initialize types
  if (warm_cluster_label_init == FALSE) {
    group_t <- sample(1:n_celltype, n, replace = TRUE)
  } else {
    group_t <- kmeans(t(log2(scRNA_data_matr+1)), n_celltype)$cluster
  }
  
  #initialize precision matrices, edge inclusion matrices and covariance matrices
  invcov_t <- array(diag(1,G), dim = c(G, G, n_celltype))
  edge_t <- array(0, dim = c(G, G, n_celltype))
  cov_t <- array(diag(1,G), dim = c(G, G, n_celltype))
  for (k in 1:n_celltype) {
    diag(invcov_t[,,k]) <- k
    diag(cov_t[,,k]) <- 1/k
  }
  
  theta_t <- t(theta_t)
  ind_zero <- t(ind_zero)
  
  #initialize mu_t
  mu_t <- NULL
  for (k in 1:n_celltype) {
    mu_t <- cbind(mu_t, rep(k,G))
  }
  
  #set gam
  gam = hyperparameters_conc
  
  #initialize pi_t
  pi_t <- update_pi_R(group_t, gam, n_celltype)
  
  #initialize lambda_t
  lambda0_t <- rnorm(G, mean=hyperparameters[5], sd=hyperparameters[7])
  lambda1_t <- rnorm(G, mean=hyperparameters[6], sd=hyperparameters[8])
  
  #set spike-slab prior
  ssp_v0 <- hyperparameters[11]
  ssp_v1 <- hyperparameters[12]
  ssp_l <- hyperparameters[13]
  ssp_pi <- hyperparameters[14]
  
  
  ###############################################################
  #######################  Gibbs Sampler ########################
  ###############################################################
  num_saved <- num_iterations - num_burnin
  
  Result <- MCMC_full(num_iterations, num_saved, theta_t, ind_zero, mu_t, 
                      invcov_t, cov_t, edge_t, group_t, lambda0_t, lambda1_t, 
                      pi_t, gam, G, n, n_celltype, ssp_v0, ssp_v1, ssp_l, ssp_pi,
                      epsilon_theta = hyperparameters[1], num_step_theta = hyperparameters[2], 
                      eta_mu = hyperparameters[3], tau_sq_mu = hyperparameters[4],
                      lam0_0= hyperparameters[5], lam1_0 = hyperparameters[6], sigma2_lam0= hyperparameters[7], 
                      sigma2_lam1= hyperparameters[8], epsilon_lam = hyperparameters[9],
                      num_step_lam = hyperparameters[10], 
                      iter_save = collect_post_sample, n_threads = num_threads, iter_print = print_per_iteration, 
                      class_print = print_label)
  Result$theta <- t(Result$theta)
  
  cell_type_adj_matr <- array(0,c(G,G,dim(Result$cell_type_edge_post_prob)[3]))
  cell_type_adj_matr[Result$cell_type_edge_post_prob >= inclusion_thr] <- 1
  Result$cell_type_adj_matr <- cell_type_adj_matr
  
  return(Result)
}



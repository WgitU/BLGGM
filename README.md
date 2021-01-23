## BLGGM: Estimating Heterogeneous Gene Regulatory Networks from Zero-Inflated Single-Cell Expression Data
Single-cell sequencing technologies can elucidate the gene-gene relationship at an unprecedented resolution. However, the expression data from single cells are often zero-inflated and heterogeneous, making commonly used Gaussian graphical models infeasible to correctly estimate gene regulatory networks. We proposed a Bayesian latent mixture Gaussian graphical model to explicitly account for the data heterogeneity and zero-inflation. The R package BLGGM (Bayesian Latent Gaussian Graph Mixture), designed for operating systems Windows and Linux, implements the proposed method to cluster cells and obtain cellular gene regulatory networks simultaneously. Package BLGGM employs a hybrid Markov chain Monte Carlo algorithm to perform posterior inference in the Bayesian framework.

The __CODE__ that can reproduce the results in the paper "Estimating Heterogeneous Gene Regulatory Networks from Zero-Inflated Single-Cell Expression Data" can be downloaded through the link https://drive.google.com/file/d/1aNaVz9Q4AEuoCNgH4ttgQ5S1sTfwB7eZ/view?usp=sharing.

## Prerequisites and Installation

1. R version >= 3.6.
2. R packages: Rcpp (>= 1.0.3), RcppArmadillo (>= 0.9.850.1.0), stats, MASS.
3. Install the package BLGGM.
```
devtools::install_github("WgitU/BLGGM")
```

## Example Code

``` {r, eval=FALSE}
library(BLGGM)

#import example data
data(example_data)

#gene number
nrow(scRNA_data_matr)

#cell number
ncol(scRNA_data_matr)

#cell type number
n_celltype <- dim(precision_matr)[3]

#run BLGGM
set.seed(20201116)
t1 <- Sys.time()
Result <- BLGGM(scRNA_data_matr, n_celltype, num_iterations = 10000, num_threads = 10)
t2 <- Sys.time()

#time cost
print(t2 - t1)

#Compared with true cell type labels
cell_table <- table(Result$cell_labels, cell_label_truth)
cell_table

#The following shows the Frobenius norms of precision matrix within each cell type
sqrt(sum((Result$cell_type_precision_matr[,,which.max(cell_table[,1])] - precision_matr[,,1])^2))
sqrt(sum((Result$cell_type_precision_matr[,,which.max(cell_table[,2])] - precision_matr[,,2])^2))
```
or you can simply run
``` {r, eval=FALSE}
library(BLGGM)
example(BLGGM)
```
## Remarks
* This package applies openmp to parallel computing. 
* This package can be downloaded and run in Windows and Linux. However, as Mac OS does not support openmp, the package temporarily does not support Mac OS.
* If you have any questions regarding this package, please contact Qiuyu Wu at w.qy@ruc.edu.cn.


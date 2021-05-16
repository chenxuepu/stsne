#' Supervised learning for t-Distributed Stochastic Neighbor Embedding
#'
#' @param X matrix; Data matrix (each row is an observation, each column is a variable)
#' @param index integer matrix; Each row contains the identity of the nearest neighbors for each observation
#' @param distance numeric matrix; Each row contains the distance to the nearest neighbors in \code{index} for each observation
#' @param dims integer; Output dimensionality (default: 2)
#' @param initial_dims integer; the number of dimensions that should be retained in the initial PCA step (default: 50)
#' @param perplexity numeric; Perplexity parameter (should not be bigger than 3 * perplexity < nrow(X) - 1, see details for interpretation)
#' @param theta numeric; Speed/accuracy trade-off (increase for less accuracy), set to 0.0 for exact TSNE (default: 0.5)
#' @param check_duplicates logical; Checks whether duplicates are present. It is best to make sure there are no duplicates present and set this option to FALSE, especially for large datasets (default: TRUE)
#' @param pca logical; Whether an initial PCA step should be performed (default: TRUE)
#' @param partial_pca logical; Whether truncated PCA should be used to calculate principal components (requires the irlba package). This is faster for large input matrices (default: FALSE)
#' @param max_iter integer; Number of iterations (default: 1000)
#' @param verbose logical; Whether progress updates should be printed (default: global "verbose" option, or FALSE if that is not set)
#' @param ... Other arguments that can be passed to Rtsne
#' @param is_distance logical; Indicate whether X is a distance matrix (default: FALSE)
#' @param Y_init matrix; Initial locations of the objects. If NULL, random initialization will be used (default: NULL). Note that when using this, the initial stage with exaggerated perplexity values and a larger momentum term will be skipped.
#' @param pca_center logical; Should data be centered before pca is applied? (default: TRUE)
#' @param pca_scale logical; Should data be scaled before pca is applied? (default: FALSE)
#' @param normalize logical; Should data be normalized internally prior to distance calculations with \code{\link{normalize_input}}? (default: TRUE)
#' @param stop_lying_iter integer; Iteration after which the perplexities are no longer exaggerated (default: 250, except when Y_init is used, then 0)
#' @param mom_switch_iter integer; Iteration after which the final momentum is used (default: 250, except when Y_init is used, then 0)
#' @param momentum numeric; Momentum used in the first part of the optimization (default: 0.5)
#' @param final_momentum numeric; Momentum used in the final part of the optimization (default: 0.8)
#' @param eta numeric; Learning rate (default: 200.0)
#' @param exaggeration_factor numeric; Exaggeration factor used to multiply the P matrix in the first part of the optimization (default: 12.0)
#' @param num_threads integer; Number of threads to use when using OpenMP, default is 1. Setting to 0 corresponds to detecting and using all available cores
#' @param weights vector;the weight for cost
#' @param col_weights vector;the weight for column. Note that when using this, the pca_scale will Mandatory designation as FALSE.Note that when using this and pca = FALSE,the normalize will Mandatory designation as FALSE. (default: NULL)
#' @param only_r logical;if TRUE will use r code only,and theta will be set as 0.0.if FALSE will use c++ code to do (default: FALSE)
#' @param r numeric;a param the can regulation density of point , if r != 2 ,then theta will be set as 0.0. (default: 2)
#'
#' @return List with the following elements:
#' \item{Y}{Matrix containing the new representations for the objects}
#' \item{N}{Number of objects}
#' \item{origD}{Original Dimensionality before TSNE (only when \code{X} is a data matrix)}
#' \item{perplexity}{See above}
#' \item{theta}{See above}
#' \item{costs}{The cost for every object after the final iteration}
#' \item{itercosts}{The total costs (KL-divergence) for all objects in every 50th + the last iteration}
#' \item{stop_lying_iter}{Iteration after which the perplexities are no longer exaggerated}
#' \item{mom_switch_iter}{Iteration after which the final momentum is used}
#' \item{momentum}{Momentum used in the first part of the optimization}
#' \item{final_momentum}{Momentum used in the final part of the optimization}
#' \item{eta}{Learning rate}
#' \item{exaggeration_factor}{Exaggeration factor used to multiply the P matrix in the first part of the optimization}
#' \item{weights}{the weight for cost}
#' \item{r}{a param the can regulation density of point}
#'
#' @importFrom stats model.matrix na.fail prcomp rnorm dist var
#' @importFrom magrittr %>%
#'
#' @references Maaten, L. Van Der, 2014. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research, 15, p.3221-3245.
#' @references van der Maaten, L.J.P. & Hinton, G.E., 2008. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research, 9, pp.2579-2605.
#'
#'
#' @examples
#' iris_unique <- unique(iris) # Remove duplicates
#' iris_matrix <- as.matrix(iris_unique[,1:4])
#'
#' # Set a seed if you want reproducible results
#' set.seed(42)
#' tsne_out <- stsne(iris_matrix,pca=FALSE,perplexity=30,theta=0.0) # Run TSNE
#'
#' # Show the objects in the 2D tsne representation
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#'
#' # data.frame as input
#' tsne_out <- stsne(iris_unique,pca=FALSE, theta=0.0)
#'
#' # Using a dist object
#' set.seed(42)
#' tsne_out <- stsne(dist(normalize_input(iris_matrix)), theta=0.0)
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#'
#' set.seed(42)
#' tsne_out <- stsne(as.matrix(dist(normalize_input(iris_matrix))),theta=0.0)
#' plot(tsne_out$Y,col=iris_unique$Species, asp=1)
#'
#' # Supplying starting positions (example: continue from earlier embedding)
#' set.seed(42)
#' tsne_part1 <- stsne(iris_unique[,1:4], theta=0.0, pca=FALSE, max_iter=350)
#' tsne_part2 <- stsne(iris_unique[,1:4], theta=0.0, pca=FALSE, max_iter=650, Y_init=tsne_part1$Y)
#' plot(tsne_part2$Y,col=iris_unique$Species, asp=1)
#' \dontrun{
#' # Fast PCA and multicore
#'
#' tsne_out <- stsne(iris_matrix, theta=0.1, partial_pca = TRUE, initial_dims=3)
#' tsne_out <- stsne(iris_matrix, theta=0.1, num_threads = 2)
#' }
#' @useDynLib stsne, .registration = TRUE
#' @import Rcpp
#' @importFrom stats model.matrix na.fail prcomp
#'
#' @export
stsne <- function (X, ...) {
  UseMethod("stsne", X)
}




#' @describeIn stsne Default Interface
#' @export
stsne.default <- function(X, dims=2, initial_dims=50,
                          perplexity=30, theta=0.5,
                          check_duplicates=TRUE,
                          pca=TRUE, partial_pca=FALSE, max_iter=1000,verbose=getOption("verbose", FALSE),
                          is_distance=FALSE, Y_init=NULL,
                          pca_center=TRUE, pca_scale=FALSE, normalize=TRUE,
                          stop_lying_iter=ifelse(is.null(Y_init),250L,0L),
                          mom_switch_iter=ifelse(is.null(Y_init),250L,0L),
                          momentum=0.5, final_momentum=0.8,
                          eta=200.0, exaggeration_factor=12.0, num_threads=1,weights = rep(1,NROW(X)),col_weights = NULL,
                          only_r = FALSE,r = 2, ...){

  if (!is.logical(is_distance)) { stop("is_distance should be a logical variable")}
  if (!is.matrix(X)) { stop("Input X is not a matrix")}
  if (is_distance & !(is.matrix(X) & (nrow(X)==ncol(X)))) { stop("Input is not an accepted distance matrix") }
  if (!(is.logical(pca_center) && is.logical(pca_scale)) ) { stop("pca_center and pca_scale should be TRUE or FALSE")}
  if (!is.wholenumber(initial_dims) || initial_dims<=0) { stop("Incorrect initial dimensionality.")}
  if (!is.numeric(r)) { stop("r should be a numeric variable")}
  if (only_r | r != 2) {theta = 0.0}
  tsne.args <- .check_tsne_params(nrow(X), dims=dims, perplexity=perplexity, theta=theta, max_iter=max_iter, verbose=verbose,
                                  Y_init=Y_init, stop_lying_iter=stop_lying_iter, mom_switch_iter=mom_switch_iter,
                                  momentum=momentum, final_momentum=final_momentum, eta=eta, exaggeration_factor=exaggeration_factor,weights = weights,only_r = only_r,r = r)
  if(!is.null(col_weights)&length(col_weights)!=ncol(X)) { stop("number of col_weights need equal to column of X") }
  if(!is.null(col_weights)) pca_scale <- FALSE
  if(!is.null(col_weights)&!pca) normalize <- FALSE
  # Check for missing values
  X <- na.fail(X)

  if (!is_distance) {
    if(!is.null(col_weights)){
      X <- addColWeight(X,col_weights)
    }
    if (pca) {
      if(verbose) cat("Performing PCA\n")
      if(partial_pca){
        if (!requireNamespace("irlba", quietly = TRUE)) {stop("Package \"irlba\" is required for partial PCA. Please install it.", call. = FALSE)}
        X <- irlba::prcomp_irlba(X, n = initial_dims, center = pca_center, scale = pca_scale)$x
      }else{
        if(verbose & min(dim(X))>2500) cat("Consider setting partial_pca=TRUE for large matrices\n")
        X <- prcomp(X, retx=TRUE, center = pca_center, scale. = pca_scale, rank. = initial_dims)$x
      }
    }
    if (check_duplicates) {
      if (any(duplicated(X))) { stop("Remove duplicates before running TSNE.") }
    }
    if (normalize) {
      if(only_r){
        X <- normalize_input_r(X)
      } else {
        X <- normalize_input(X)
      }
    }
    if(!only_r){X <- t(X) }
  }else{
    # Compute Squared distance if we are using exact TSNE
    if (theta==0.0) {
      X <- X^2
    }
  }

  if(only_r){
    out <- RTsne_R(X=X,is_distance=is_distance,args=tsne.args)
    info <- list(N=nrow(X))
  } else {
    out <- do.call(Rtsne_cpp, c(list(X=X, distance_precomputed=is_distance, num_threads=num_threads), tsne.args))
    out$Y <- t(out$Y)
    info <- list(N=ncol(X))
  }
  if (!is_distance) { out$origD <- ncol(X) } # 'origD' is unknown for distance matrices.
  out <- c(info, out, .clear_unwanted_params(tsne.args))
  class(out) <- c("stsne","list")
  out
}


#' @describeIn stsne on given dist object
#' @export
stsne.dist <- function(X,...,is_distance=TRUE) {
  X <- as.matrix(na.fail(X))
  stsne(X, ..., is_distance=is_distance)
}

#' @describeIn stsne on data.frame
#' @export
stsne.data.frame <- function(X,...) {
  X <- model.matrix(~.-1,na.fail(X))
  stsne(X, ...)
}


RTsne_R <- function(X,is_distance,args){
  verbose <- args$verbose
  if(args$init){
    Y <- args$Y_in
    if (verbose) cat("Using user supplied starting positions\n");
  } else{
    Y <- matrix(rnorm(nrow(X)*args$no_dims,sd = 10^-4),ncol = args$no_dims)
  }
  if (verbose) cat("Using no_dims =", args$no_dims, ", perplexity =",args$perplexity,"\n");
  if (verbose) cat("Computing input similarities...\n");
  start <- Sys.time()
  if (verbose) cat("Symmetrizing...\n");
  P <- computeGaussianPerplexity(X,is_distance,args$perplexity) %>%
    Symmetrize()
  end <- Sys.time()
  if (verbose) cat("Done in ",as.character.Date(end - start),"\nLearning embedding...\n");
  trainIterations(P,Y,args)
}




trainIterations <- function(P,Y,args){
  verbose <- args$verbose
  P <- P*args$exaggeration_factor
  start <- Sys.time()
  total_time <- 0
  momentum <- args$momentum
  uY <- matrix(0,nrow = nrow(Y),ncol = ncol(Y))
  gains <- matrix(1,nrow = nrow(Y),ncol = ncol(Y))
  for(i in 1:args$max_iter){
    if(i == args$stop_lying_iter) P <- P/args$exaggeration_factor
    if(i == args$mom_switch_iter) momentum = args$final_momentum;
    DY <- computeExactGradient(P,Y,args)
    gains <- ifelse(sign(DY)!=sign(uY),gains+0.2,gains*0.8)
    gains[gains<0.01] <- 0.01
    uY <- momentum*uY - args$eta*DY*gains
    Y <- Y+uY
    Y <- zeroMean(Y)
    if((i>1&i%%50==0)|i==args$max_iter){
      end <- Sys.time()
      C <- evaluateError(P,Y,args)
      if(i == 1) {
        if (verbose) cat("Iteration ",i,": error is ",C,"\n");
      }
      else {
        total_time <-  total_time + end - start
        if (verbose) cat("Iteration ",i,": error is ",C," (50 iterations ",as.character.Date(end - start),")\n")
      }
      start <- Sys.time()
    }
  }
  end <- Sys.time()
  total_time <-  total_time + end - start
  cost <- getCost(P,Y,args)
  if (verbose) cat("Fitting performed in ",as.character.Date(total_time),"\n")
  out <- list(Y = Y,costs = cost)

}


computeExactGradient <- function(P,Y,args){
  n <- nrow(Y)
  dc <- matrix(0,nrow = n,ncol = ncol(Y))
  y_sum <- apply(Y^2, 1, sum)
  Q <- 1/(1 + y_sum +    sweep(-2 * Y %*% t(Y),2, -t(y_sum)))
  diag(Q) <- 0
  sum_Q <- sum(Q)
  M <- (P-(Q/sum_Q))*Q
  for(i in 1:ncol(Y)){
    dc[,i] <- (M %*% (matrix(Y[,i],n,n,byrow = T)-matrix(Y[,i],n,n))) %>% diag()
  }
  dc <- sweep(dc,1,FUN = "*",args$weight)
  return(dc)
}

zeroMean <- function(Y){
  Y - matrix(colMeans(Y),nrow = nrow(Y),ncol = ncol(Y),byrow = T)
}



evaluateError <- function(P,Y,args){
  w <- matrix(args$weight,nrow = nrow(Y),ncol = nrow(Y))
  DD <- dist(Y) %>% as.matrix()
  DD <- DD^2
  Q <- 1/(1+DD)
  diag(Q) <- 0
  Q <- Q/sum(Q)
  C <- (w*P*log((P+1e-9)/(Q+1e-9))) %>% sum
  return(C)
}


getCost <- function(P,Y,args){
  w <- matrix(args$weight,nrow = nrow(Y),ncol = nrow(Y))
  DD <- dist(Y) %>% as.matrix()
  DD <- DD^2
  Q <- 1/(1+DD)
  diag(Q) <- 0
  Q <- Q/sum(Q)
  Cost <- (w*P*log((P+1e-9)/(Q+1e-9))) %>% rowSums()
  return(Cost)
}

computeGaussianPerplexity <- function(X,is_distance,perplexity){
  if(!is_distance){
    D <- dist(X) %>% as.matrix()
    D <- D^2
  }
  else{
    D <- X
  }
  diag(D) <- NA
  apply(D, 1, function(x){
    found <- FALSE
    beta <- 1.0
    min_beta <- -Inf
    max_beta <- Inf
    tol <- 1e-5
    sum_p <- 0
    iter <- 0
    while(!found&iter<200){
      p <- exp(-beta*x)
      # x[is.na(x)] <- 0
      p[is.na(p)] <- 0
      sum_p <- sum(p)
      H <- sum(beta*x*p,na.rm = TRUE)/sum_p+log(sum_p)
      Hdiff <- H-log(perplexity)
      if(abs(Hdiff) < tol){
        found <- TRUE
      }else{
        if(Hdiff > 0){
          min_beta <- beta
          if(max_beta == Inf){
            beta <- beta*2
          }else{
            beta <- (beta + max_beta)/2
          }
        }else{
          max_beta <- beta
          if(min_beta == -Inf){
            beta <- beta/2
          }else{
            beta <- (beta + min_beta)/2
          }
        }
      }
      iter <- iter+1
    }
    p/sum_p
  }) %>% t()
}


Symmetrize <- function(X){
  X <- X+t(X)
  X/sum(X)
}

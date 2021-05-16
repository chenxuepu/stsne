#' help set weight for the data
#'
#' @param Y matrix,vector,data.frame; Dependent Variable from other machine learning method
#' @param X matrix; Data matrix (each row is an observation, each column is a variable)
#' @param weight_by_var logical; use variance to decide weight,if FALSE need to set Y_weight and X_weight
#' @param Y_weight numeric; the weight for Y
#' @param X_weight numeric; the weight for X
#' @param ... Other arguments that can be passed to stsne(except col_weights)
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
#' @export
Sutsne <- function(Y,X,weight_by_var = TRUE,Y_weight = NULL,X_weight = NULL,...){
  if(!is.matrix(X)&!is.data.frame(X)){
    stop("X should be a matrix or data.frame")
  }
  if(!is.matrix(Y)&!is.data.frame(Y)&!is.vector(Y)){
    stop("Y should be a matrix ,vector or data.frame")
  }
  if(NROW(Y)!=NROW(X)) {stop("X and Y should have the same row")}
  if(!is.null(Y_weight)&!is.null(X_weight)) weight_by_var = FALSE
  if(weight_by_var){
    w <- weightVar(Y,X)
  }else{
    if(is.null(Y_weight)|is.null(X_weight)) {stop("should be set the weight") }
    w <- weight_by_set(Y = Y,X = X,Y_weight = Y_weight,X_weight = X_weight)
  }
  data <- cbind(Y,X)

  stsne(X = data,col_weights = w)
}




weightVar <- function(Y,X){
  X_sd <- apply(X,2,var) %>% mean() %>% sqrt()
  if(is.vector(Y)){
    c(X_sd*ncol(X),rep(1,ncol(X)))
  }else{
    c(rep(X_sd,ncol(Y))*ncol(X)/ncol(Y),rep(1,ncol(X)))
  }
}


weight_by_set <- function(Y,X,Y_weight,X_weight){
  if(is.vector(Y)){
    c(Y_weight,rep(X_weight,ncol(X)))
  }else{
    c(rep(Y_weight,ncol(Y)),rep(X_weight,ncol(X)))
  }
}



#' use pca or other method to set Initial locations ,then the tsne can predict.
#'
#' @param data matrix; Data matrix (each row is an observation, each column is a variable)
#' @param dims integer; Output dimensionality (default: 2)
#' @param use_pca logical; If TRUE,will use pca to set Initial locations(default: TRUE)
#' @param Y_init matrix; If use other method to  set Initial locations of the objects,given the Initial locations in here. If use pca ,keep if NULL (default: NULL).
#' @param ... Other arguments that can be passed to stsne(except dims,check_duplicates and Y_init )
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
#' \item{Y_init}{the MDS value to Y_init}
#' \item{use_pca}{whether use pca to set Initial locations}
#' \item{pca_coef}{we use pca to measuring MDS, we keep the value for predict}
#' \item{center}{the data center, predict needs to use it}
#' @export
train_stsne <- function( data, dims = 2, use_pca = TRUE, Y_init=NULL,...){
  ## the function is use by myself , so it has not check anything
  if(!is.null(Y_init)){
    use_pca = FALSE
  }
  if(use_pca){
    pca <- stats::prcomp(data)
    Y_init <- as.matrix(pca$x[,1:dims])
  }
  if(!use_pca&is.null(Y_init)) {stop("need use pca or Y_init")}


  tsne <- stsne(data,dims = dims,check_duplicates = FALSE,Y_init = Y_init,... = ...)
  tsne[["Y_init"]] <- Y_init
  tsne[["use_pca"]] <- use_pca
  if(use_pca){
    tsne[["pca_coef"]] <- as.matrix(pca$rotation[,1:dims])
    tsne[["center"]] <- pca$center
  }
  class(tsne) <- "tsne"
  tsne
}






#' predict T-SNE
#'
#' @param object list; a list from \link{train_stsne}
#' @param data matrix; a data want to predict
#' @param k numeric; how many neighbor to use (default: 5)
#' @param Y_init matrix; If use other method to  set Initial locations of the objects,given the Initial locations in here. If use pca ,keep if NULL (default: NULL).
#' @param ... Other arguments that can use.
#'
#' @return
#' a dataframe
#' @export
predict.tsne <- function(object, data, k = 5 , Y_init=NULL,...){
  ## the function is use by myself , so it has not check anything
  if(!("tsne" %in% class(object))){
    stop("class is not TSNE")
  }
  if(object$use_pca){
    data_mean <- matrix(object$center,nrow = nrow(data),ncol = ncol(data),byrow = T)
    data_Y_init <- as.matrix(data-data_mean) %*% object$pca_coef
  }else{
    if(is.null(Y_init)) {stop("need given Y_init")}
    data_Y_init <- Y_init
  }
  RS <- object$Y-object$Y_init
  dims <- ncol(object$Y)

  data_predict <- matrix(nrow = nrow(data),ncol = dims) %>%
    as.data.frame()

  for(i in 1:nrow(data)){
    data_dist_order <- (object$Y_init-matrix(data_Y_init[i,],ncol = dims,nrow = nrow(object$Y_init),byrow = T))^2 %>% rowSums() %>% order()
    index <- data_dist_order[1:k]
    move <- RS[index,] %>% colMeans()
    data_predict[i,] <- data_Y_init[i,]+move
  }
  data_predict
}

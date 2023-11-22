
#' plot.ART
#' @description Plot a 2 dimensional Fuzzy ART category distribution
#' @param net An ART network
#' @param id The numeric id of the module to be plotted. The base module ART a (default) has an id = 0. The ids of the higher order module are incremented.
#' @param .data The data frame used for plotting the data. It must have 2 columns corresponding to the x and y values, respectively.
#' @param x The variable to plot along the x axis. If it is not specified, then the first column of the data and the weight matrix
#' will be used.
#' @param y The variable to plot along the y axis. If it is not specified, then the second column of the data and the weight matrix
#' will be used.
#' @return A ggplot
#' @importFrom ggplot2 ggplot geom_point aes_
#' @importFrom rlang sym
#' @export
plot.ART <- function(net, id = 0, .data, x = NULL, y = NULL){
  if (is.matrix(.data)){
    .data <- tryCatch(
      as.data.frame(.data),
      error = function(e){
        e
      }
    )
  }
  if (inherits(.data, "error")){
    stop(.data$message)
  }
  module <- getModuleById(net, id)

  cols <- colnames(.data)
  if (is.null(x)) x <- cols[1]
  if (is.null(y)) y <- cols[2]
  
  ggplot() + geom_point(data = .data[, c(x, y)], aes_(sym(x), sym(y))) +
    drawWeight(structure(module$w, class = append(class(module$w), getRule(net))),
               x = x, y = y)

}
#' plot.ARTMAP
#' @description Plot a 2 dimensional Fuzzy ARTMAP category distribution
#' @param net An ARTMAP network
#' @param .data The data frame used for plotting the data. It must have 2 columns specifying x and y values.
#' @param x The variable to plot along the x axis. If it is not specified, then the first column of the data and the weight matrix
#' will be used.
#' @param y The variable to plot along the y axis. If it is not specified, then the second column of the data and the weight matrix
#' will be used.
#' @param classes The class labels associated with the data points.
#' @param dummyCodeMap The dummy code rule that maps each class label to its dummy code.
#' @return A ggplot
#' @importFrom ggplot2 ggplot geom_point aes_ sym
#' @export
plot.ARTMAP <- function(net, .data, x = NULL, y = NULL, classes = NULL, dummyCodeMap = NULL){

  if (!is.data.frame(.data)){
    .data <- tryCatch(
      as.data.frame(.data),
      error = function(e){
        e
      }
    )
  }
  if (inherits(.data, "error")){
    stop(.data$message)
  }

  if (is.null(classes)){
    stop("The class labels are missing.")
  }
  # organize the data for plotting
  if (is.null(dummyCodeMap)){
    .data <- cbind(.data, classes = classes)
  } else{
    .data <- cbind(.data, classes = decode(classes, dummyCodeMap))
  }
  #if (isSimplified(net)){
  #  .data <- cbind(.data, classes = classes)
  #} else{
    # convert the dummy codes to the class labels
  #  if (is.null(dummyCodeMap)){
  #    stop("You need to supply the dummy code mapping to convert the dummy codes into class labels.")
  #  }
  #  .data <- cbind(.data, classes = decode(classes, dummyCodeMap))
  #}
  cols <- colnames(.data)
  if (is.null(x)) x <- cols[1]
  if (is.null(y)) y <- cols[2]

  # organize the weights for plotting
  module <- getModuleById(net, id = 0) # always plot ART a
  labels <- NULL
  if (!isSimplified(net)){
    labels <- decode(net$mapfield$w, dummyCodeMap)
  } else{
    labels <- net$mapfield$w
  }

  ggplot() + geom_point(data = .data, aes_(x = sym(cols[1]), y = sym(cols[2]), color = sym(cols[3]))) +
    drawWeight(structure(module$w, class = append(class(module$w), getRule(net))), 
               x = x, y = y, labels)
}

#' plot.TopoART
#' @description Plot a 2 dimensional Fuzzy TopoART category distribution
#' @param net A TopoART network
#' @param id The numeric id of the module to be plotted. The first module a (default) has an id = 0. Module b has an id = 1. The ids of the higher order modules are incremented.
#' @param .data The data frame used for plotting the data. It must have 2 columns corresponding to the x values and y values, respectively.
#'              If .data has more than 2 columns, only the first 2 columns will be used.
#' @param x The variable to plot along the x axis. If it is not specified, then the first column of the data and the weight matrix
#' will be used.
#' @param y The variable to plot along the y axis. If it is not specified, then the second column of the data and the weight matrix
#' will be used.
#' @return A ggplot
#' @importFrom ggplot2 ggplot geom_point aes_ sym
#' @export
plot.TopoART <- function(net, id = 0, .data, x = NULL, y = NULL){
  if (is.matrix(.data)){
    .data <- tryCatch(
      as.data.frame(.data),
      error = function(e){
        e
      }
    )
  }
  if (inherits(.data, "error")){
    stop(.data$message)
  }
  module <- getModuleById(net, id)
  categories <- getTopoClustersCategories(module)
  
  cols <- colnames(.data)
  if (is.null(x)) x <- cols[1]
  if (is.null(y)) y <- cols[2]
  
  ggplot() + geom_point(data = .data, aes_(sym(x), sym(y))) +
    drawWeight(structure(module$w, class = append(class(module$w), getRule(net))), 
               x = x, y = y, categories)
}

#' drawWeight
#' @description Generic function for plotting weights
#' @param w The weight object
#' @param x The x axis variable
#' @param y The y axis variable
#' @param ... Additional arguments to be passed
#' @export
drawWeight <- function(w, x, y, ...){
  UseMethod("drawWeight", w)
}

#' drawWeight.hypersphere
#' @description Plot 2D circles
#' @param w The data frame with the first 2 columns being x and y coordinates of the weight, and the third column being the radii.
#' @param classLabels The vector of class labels associated with the weights in F2.
#' @importFrom ggforce geom_circle
#' @export
drawWeight.hypersphere <- function(w, x, y, classLabels = NULL){
  if (!is.null(classLabels) && length(classLabels) != nrow(w))
    stop("The length of the class labels must be the same as the number of rows in the w matrix.")

  w <- data.frame(x = w[,x], y = w[,y], R = w[,"R"])
  if (is.null(classLabels)){
    w <- cbind(w, label = rep("Cluster", nrow(w)))
    g <- geom_circle(data = w,
                     aes(x0 = x, y0 = y, r = R, color = label, fill = label),
                     alpha = 0.1)
  } else {
    w <- cbind(w, Label = factor(classLabels))
    g <- geom_circle(data = w,
                     aes(x0 = x, y0 = y, r = R, color = Label, fill = Label),
                     alpha = 0.1)
  }
  return (g)
}
#' drawWeight.fuzzy
#' @description Draw the 2D rectangular weights created by the fuzzy rule
#' @param w The weight matrix
#' @param x The x axis variable
#' @param y The y axis variable
#' @param classLabels The vector of class labels associated with all weights in F2
#' @return A ggplot
#' @importFrom ggplot2 geom_rect aes
#' @export
drawWeight.fuzzy <- function(w, x, y, classLabels = NULL){
  if (!is.null(classLabels) && length(classLabels) != nrow(w))
    stop("The length of the class labels must be the same as the number of rows in the w matrix.")
  
  x_c <- paste0(x, "_c")
  y_c <- paste0(y, "_c")
  
  w <- data.frame(ux = w[,x], uy = w[,y],
                  vx = 1-w[,x_c], vy = 1-w[,y_c])
  if (is.null(classLabels)){
    w <- cbind(w, label = rep("Cluster", nrow(w)))
    g <- geom_rect(data = w,
                   aes(xmin = ux, xmax = vx, ymin = uy, ymax = vy, color = label, fill = label),
                   alpha = 0.1)
  } else {
    w <- cbind(w, Label = factor(classLabels))
    g <- geom_rect(data = w,
                   aes(xmin = ux, xmax = vx, ymin = uy, ymax = vy, fill = Label, color = Label),
                   alpha = 0.1)
  }
  return (g)
}

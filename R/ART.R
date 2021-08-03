
#' Create an ART Network
#' @description Create a new ART object
#' @param rule The activation / match / learning rule to use. The choices available are fuzzy and hypersphere.
#' @param dimension The number of dimension/features in the data.
#' @param vigilance The vigilance parameter. Must be between 0 and 1.
#' @param learningRate The learning rate. Must be between 0 and 1.
#' @return ART returns an ART object.
#' @export
ART <- function(rule = c("fuzzy", "hypersphere"), dimension, vigilance = 0.7, learningRate = 1.0, maxEpochs = 10){

  rule <- match.arg(rule)

  art <- .ART( dimension, 1, vigilance, learningRate)

  attr(art, "rule") <- rule

  return (art)
}
#' Create a Topological ART Network
#' @description Create a new TopoART object
#' @param rule The activation / match / lerning rule to use. The choices available are fuzzy and hypersphere.
#' @param dimension The number of dimension/features in the data
#' @param vigilance The vigilance parameter. Must be between 0 and 1.
#' @param learningRate1 The learning rate when this neuron has the highest activation. Must be between 0 and 1.
#' @param learningRate2 The learning rate when this neuron has the second highest activation. Must be between 0 and 1.
#' @param tau The number of learning cycles before the F2 neurons with counts below the threshold are removed. Default is 100.
#' @param phi The count threshold. The F2 neuron is kept when the count >= phi, removed when the count < phi.
#' @param maxEpochs The maximum number of epochs. Default is 20.
#' @return TopoART returns a TopoART object.
#' @export
TopoART <- function(rule = c("fuzzy", "hypersphere"), dimension, vigilance = 0.7, learningRate1 = 1.0, learningRate2 = 0.6, tau = 100, phi = 6, maxEpochs = 20){

  rule <- match.arg(rule)

  topoART <- .TopoART(dimension, 2, vigilance, learningRate1, learningRate2, tau, phi, maxEpochs = maxEpochs)

  attr(topoART, "rule") <- rule

  return (topoART)
}

#' Create an ARTMAP Network
#' @description Create a new ARTMAP object
#' @param rule The activation / match / learning rule to use. The choices available are fuzzy and hypersphere.
#' @param dimension The number of dimension/features in the data
#' @param vigilance The vigilance parameter. Must be between 0 and 1.
#' @param learningRate The learning rate. Must be between 0 and 1.
#' @param maxEpochs The maximum number of epochs. Default is 10.
#' @param simplified Logical. Whether to run the simplified version of ARTMAP. Default is FALSE.
#' @return ARTMAP returns an ARTMAP object.
#' @export
ARTMAP <- function(rule = c("fuzzy", "hypersphere"), dimension, vigilance = 0.7, learningRate = 1.0, maxEpochs = 10, simplified = TRUE){
  rule <- match.arg(rule)

  artmap <- .ARTMAP(numFeatures = dimension, vigilance = vigilance, learningRate = learningRate, maxEpochs = maxEpochs, simplified = simplified)

  attr(artmap, "rule") <- rule

  return (artmap)
}

#' Train
#' @description A generic function for training an ART network.
#' @param network An ART or ARTMAP object
#' @export
train <- function(network, ...){
  UseMethod("train", network)
}

#' Train an ART network
#' @description The ART training method
#' @param network An ART  object
#' @param .data The data used for training.
#' @export
train.ART <- function(network, .data){
  .trainART(network, .data)
}

#' Train a Topological ART Network
#' @description The TopoART training method
#' @param network An TopoART  object
#' @param .data The data used for training.
#' @export
train.TopoART <- function(network, .data){
  .topoTrain(network, .data)
}

#' Train an ARTMAP Network
#' @description The ARTMAP training method
#' @param network An ARTMAP object
#' @param .data The data used for training.
#' @param classLabels The classes for each data point. They are numeric values.
#' @param dummyClasses The classes for each data point. These are dummy codes for the labels and must be binary.
#' @param simplified Logical. Whether to use the simplified ART network. Default is FALSE.
#' @export
train.ARTMAP <- function(network, .data, classLabels = NULL, dummyClasses = NULL){
  .trainARTMAP(network, .data, classLabels, dummyClasses)
}

#' ART Prediction
#' @description The ART prediction/classification method
#' @param network An ART object
#' @param id The id of the module
#' @param .data The data used for prediction/testing. The data must be normalized between 0 and 1.
#' @return Returns a list containing the predicted F2 categories.
#' @export
predict.ART <- function(network, id, .data){
  .predictART(network, id, .data)
}

#' Topological ART Prediction
#' @description The TopoART prediction/classification method
#' @param network A TopoART object
#' @param id The id of the module
#' @param .data The data used for prediction. The data must be normalized between 0 and 1.
#' @return Returns a list containing the predicted F2 categories and the linked clusters.
#' @export
predict.TopoART <- function(network, id, .data){
  .topoPredict(network, id, .data)
}

#' ARTMAP Prediction
#' @description The ARTMAP prediction/classification method
#' @param network An ARTMAP object
#' @param .data The data used for training. The data must be normalized between 0 and 1.
#' @param classLabels The classes for each data point. They mustx be numeric values.
#' @param dummyClasses The classes for each data point. These are dummy codes for the labels and must be binary.
#' @return Returns a list containing three items: 1. categories - the mapfield categories predicted, 2. category_a - the F2 categories predicted, and 3. matched - whether the mapfield categories predicted match the actual values.
#' @export
predict.ARTMAP <- function(network, .data, classLabels = NULL, dummyClasses = NULL){
  if (!is.matrix(.data)){
    .data <- tryCatch(
      as.matrix(.data),
      error = function(e){
        e
      }
    )
  }

  if (inherits(.data, "error")){
    stop(.data$message)
  }

  test <- FALSE
  if (!is.null(dummyClasses) || !is.null(classLabels)){
    test <- TRUE
  }

  if (!is.null(classLabels) && !is.numeric(classLabels)){
    classLabels <- as.numeric(classLabels)
  }
  .predictARTMAP(network, .data, classLabels, dummyClasses, test)
}

#' Get the Learning Rule
#' @description Get the type of the learning rule
#' @param net An ART network
#' @return The name of the rule.
#' @export
getRule <- function(net){
  return (attr(net, "rule"))
}

#' Select a Module
#' @description Get the module from the network by the id
#' @param net An ART network
#' @param id The id of the module. Either a numeric or a character/string id.
#' @return An ART module
#' @export
getModuleById <- function(net, id){

  if (is.numeric(id)){
    for (module in net$module){
      if (module$id == id)
        return (module)
    }
  } else if (is.character(id)){
    for (n in names(net$module)){
      if (n == id)
        return (net$module[[n]])
    }
  }
  stop(paste("Module", id, "does not exist in the network."))
}

#' Topological ART Clusters
#' @description Return all linked cluster categories created by a TopoART module
#' @param module A TopoART module
#' @return A vector of linked cluster categories
#' @export
getTopoClustersCategories <- function(module){
  categories <- rep(0, nrow(module$w))

  if (length(module$linkedClusters) > 0){
    module$linkedClusters <- t(module$linkedClusters)
    for (i in seq_len(length(module$linkedClusters))){
      for (j in seq_len(length(module$linkedClusters[[i]]))){
        categories[module$linkedClusters[[i]][j] + 1] <- i
      }
    }
    categories <- factor(categories)
  }

  return (categories)
}

#' Make Dummy Code
#' @description Generate the dummy codes for all the possible class labels
#' @param classLabels A vector of class labels. The labels must be numeric and unique.
#' @return A list of dummy codes with the keys being the class labels and the values being the dummy (binary) codes.
#' @export
createDummyCodeMap <- function(classLabels){
  if (is.factor(classLabels)){
    classLabels <- as.character(classLabels)
  }

  l <- length(classLabels)
  s <- seq(1, l)
  code <- list()
  if (is.numeric(classLabels)){
    for (i in seq_along(classLabels)){
      code[[as.character(classLabels[i])]] <- rep(0, l)
      code[[as.character(classLabels[i])]][s[i]] <- 1
    }
  } else if (is.character(classLabels)){
    for (i in seq_along(classLabels)){
      code[[classLabels[i]]] <- rep(0, l)
      code[[classLabels[i]]][s[i]] <- 1
    }
  }

  return (code)
}

#' Label Encoding and Decoding
#' @description Convert numeric code to binary dummy code
#' @param classLabels A vector of class labels. They must be numeric.
#' @param dummyCode A list of dummy codes created for the labels using the function dummyCode
#' @return A matrix of binary values
#' @rdname createDummyCodeMap
#' @export
encodeLabel <- function(classLabels, dummyCode){
  if (is.factor(classLabels)){
    classLabels <- as.character(classLabels)
  }
  convert <- NULL
  if (is.numeric(classLabels)){
    convert <- .encodeNumericLabel(classLabels, dummyCode)
  } else if (is.character(classLabels)){
    convert <- .encodeStringLabel(classLabels, dummyCode)
  }
  return (convert)
}

#' Decode
#' @description Convert dummy code back to the class label
#' @param dummyClasses The matrix of the dummy codes associated with the data points
#' @param dummyCode The mapping that converts the class labels to dummy code and vice versa.
#' @return A vector of class labels
#' @rdname createDummyCodeMap
#' @export
decode <- function(dummyClasses, dummyCode){
  labels <- vector(mode = "numeric", length = nrow(dummyClasses))
  for (i in seq_len(nrow(dummyClasses))){
    for (j in seq_len(length(dummyCode))){
      if (identical(dummyCode[[j]], dummyClasses[i,])){
        labels[i] <- names(dummyCode)[j]
        break
      }
    }
  }
  return (labels)
}

#' Data Normalization
#' @description Normalize each column of data to values between 0 and 1
#' @param .data A data object such as a data frame, a matrix , or an array.
#' @return A normalized data object
#' @export
normalize <- function(.data){
  for (n in seq_len(ncol(.data))){
    cmax <- max(.data[, n], na.rm = T)
    cmin <- min(.data[, n], na.rm = T)
    .data[, n] <- (.data[, n] - cmin)/(cmax - cmin)
  }
  return (.data)
}

#' Find the maximum for each column
#' @param .data A data object such as a data frame, a matrix, or an array.
#' @param na.rm Whether to remove NA from the calculation. Default is FALSE.
#' @return A vector of maximum values for each column
#' @export
colMax <- function(.data, na.rm = FALSE){
  m <- sapply(seq_len(ncol(.data)), function(k){
    max(.data[,k], na.rm = na.rm)
  })
  return (m)
}

#' Find the minimum for each column
#' @param .data A data object such as a data frame, a matrix, or an array.
#' @param na.rm Whether to remove NA from the calculation. Default is FALSE.
#' @return A vector of minimum values for each column
#' @export
colMin <- function(.data, na.rm = FALSE){
  m <- sapply(seq_len(ncol(.data)), function(k){
    min(.data[,k], na.rm = na.rm)
  })
  return (m)
}

#' Simplified ARTMAP
#' @description Check if the ARTMAP object is the simplified version.
#' @param net An ARTMAP object
#' @return isSimplified returns a logical value.
#' @rdname ARTMAP
#' @export
isSimplified <- function(net){
  if (!isARTMAP(net)){
    stop("The object is not an ARTMAP.")
  }
  return (attr(net, "simplified"))
}

#' isARTMAP
#' @description Check if the object is an ARTMAP
#' @param net An object
#' @return isARTMAP returns a logical value.
#' @rdname ARTMAP
#' @export
isARTMAP <- function(net){
  return (inherits(net, "ARTMAP"))
}

#' isART
#' @description Check if the object is an ART
#' @param net An object
#' @return isART returns a logical value.
#' @rdname ART
#' @export
isART <- function(net){
  return (inherits(net, "ART"))
}

#' isTopoART
#' @description Check if the object is a TopoART
#' @param net An object
#' @return isTopoART returns a logical value.
#' @rdname isTopoART
#' @export
isTopoART <- function(net){
  return (inherits(net, "TopoART"))
}

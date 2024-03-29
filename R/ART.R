
#' ART
#' @description Create a new ART object
#' @param numModules The number of modules in the network. Currently, it supports only 1 module.
#' @param rule The activation / match / lerning rule to use. The choices available are fuzzy.
#' @param dimension The number of dimension/features in the data.
#' @param vigilance The vigilance parameter. Must be between 0 and 1.
#' @param learningRate The learning rate. Must be between 0 and 1.
#' @param maxEpochs The maximum number of epochs to run. Default is 10.
#' @param ... Other ART model specific parameters to be initialized.
#' @export
ART <- function(numModules = 1, rule = c("fuzzy", "hypersphere", "ART1"), dimension, vigilance = 0.7, learningRate = 1.0, maxEpochs = 10, ...){
  
  p <- list(...)
  
  rule <- match.arg(rule)
 
  art <- .ART( dimension, numModules, vigilance, learningRate, maxEpochs = maxEpochs)
  
  switch(rule,
         "fuzzy" = .checkFuzzyBounds(art),
         "hypersphere" = .checkHypersphereBounds(art),
         "ART1" = .checkART1Bounds(art)
         )
  
  attr(art, "rule") <- rule
  
  return (art)
}

#' ART
#' @description Create a new TopoART object
#' @param numModules The number of modules in this network. It must be at least 2.
#' @param rule The activation / match / lerning rule to use. The choices available are fuzzy.
#' @param dimension The number of dimension/features in the data
#' @param vigilance The vigilance parameter. Must be between 0 and 1.
#' @param learningRate1 The learning rate when this neuron has the highest activation. Must be between 0 and 1.
#' @param learningRate2 The learning rate when this neuron has the second highest activation. Must be between 0 and 1.
#' @param tau The number of learning cycles before the F2 neurons with counts below the threshold are removed. Default is 100.
#' @param phi The count threshold. The F2 neuron is kept when the count >= phi, removed when the count < phi.
#' @param maxEpochs The maximum number of epochs. Default is 20.
#' @export
TopoART <- function(numModules = 2, rule = c("fuzzy", "hypersphere"), dimension, vigilance = 0.7, learningRate1 = 1.0, learningRate2 = 0.6, tau = 100, phi = 6, maxEpochs = 20){
  if (numModules < 2){
    stop("The number of modules must be at least 2.")
  }
  
  topoART <- .TopoART(dimension, numModules, vigilance, learningRate1, learningRate2, tau, phi, maxEpochs = maxEpochs)
  
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
#' @param ... Other ART model specific parameters to be initialized.
#' @return ARTMAP returns an ARTMAP object.
#' @export
ARTMAP <- function(rule = c("fuzzy", "hypersphere", "ART1"), dimension, vigilance = 0.7, learningRate = 1.0, maxEpochs = 10, simplified = TRUE, ...){
  p <- list(...)
  
  rule <- match.arg(rule)
 
  artmap <- .ARTMAP(dimension = dimension, vigilance = vigilance, learningRate = learningRate, maxEpochs = maxEpochs, simplified = simplified)

  switch(rule,
         "fuzzy" = .checkFuzzyBounds(artmap),
         "hypersphere" = .checkHypersphereBounds(artmap),
         "ART1" = .checkART1Bounds(artmap)
  )
  
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
#' @return The ART object
#' @export
train.ART <- function(network, .data){
  .trainART(network, .data)
  network <- addWeightColumnNames(network, colnames(.data))
  return (network)
}

#' Train a Topological ART Network
#' @description The TopoART training method
#' @param network An TopoART  object
#' @param .data The data used for training.
#' @return The TopoART object
#' @export
train.TopoART <- function(network, .data){
  .topoTrain(network, .data)
  network <- addWeightColumnNames(network, colnames(.data))
  return (network)
}

#' Train an ARTMAP Network
#' @description The ARTMAP training method
#' @param network An ARTMAP object
#' @param .data The data used for training.
#' @param target Either a numeric vector or a matrix. Use the vector form when running the simplified ARTMAP classification. Use the matrix 
#' form when running the standard ARTMAP classification where the target labels must be binary values. For regression which requires the 
#' standard ARTMAP, either a vector or a matrix (single column) of continuous values (normalized between 0 and 1) can be used.
#' @return The ARTMAP object
#' @export
train.ARTMAP <- function(network, .data, target){
  if (missing(target)){
    stop("The target is missing.")
  }
  if (!is.vector(target) && !is.matrix(target)){
    stop("The target must be either a vector or a matrix.")
  }
  if (!isSimplified(network)){
    if (!is.matrix(target))
      target <- as.matrix(target)
  } else{
    if (!is.vector(target)){
      stop("The simplified ARTMAP requires a vector for the target.")
    }
  }
  if (is.vector(target)){
    .trainARTMAP(network, .data, vTarget = target)
  } else{
    # it is a matrix
    .trainARTMAP(network, .data, vTarget = NULL, mTarget = target)
  }
  network <- addWeightColumnNames(network, colnames(.data))
  return (network)
}

#' Add column names to the weight matrix
#' @description Depending on the rule, add the column names in the data to the weight
#' matrix. For the fuzzy rule, the column names of the data and their complement (with _c)
#' attached to the names are added. For the hypersphere rule, the column names and the radius
#' column "R" are added. For ART1, the bottom up and top down weight columns are specified by
#' the tags "_bu" and "_td".
#' @param network The ART object
#' @param columnNames The column names to be added to the weight matrix.
#' @return The weight matrix with the column names
#' @export
addWeightColumnNames <- function(network, columnNames){
  rule <- getRule(network)
  for (i in 1:length(network$module)){
    colnames(network$module[[i]]$w) <- switch(rule,
                                              "fuzzy" = c(columnNames, paste0(columnNames, "_c")),
                                              "hypersphere" = c(columnNames, "R"),
                                              "ART1" = c(paste0(columnNames, "_bu"), paste0(columnNames, "_td"))
    )
  }
  return (network)
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
#' @param target Either a numeric vector, a matrix, or NULL. Use the vector form when running the simplified ARTMAP classification. Use the matrix 
#' form when running the standard ARTMAP classification where the target labels must be binary values. For regression which requires the 
#' standard ARTMAP, either a vector or a matrix (single column) of continuous values (normalized between 0 and 1) can be used. If it is NULL, then
#' only the predictions are done.
#' @return Returns a list containing three items: 1. categories - the mapfield categories predicted, 2. category_a - the F2 categories predicted, and 3. matched - whether the mapfield categories predicted match the actual values.
#' @export
predict.ARTMAP <- function(network, .data, target = NULL){
  
  if (!is.matrix(.data)){
    .data <- as.matrix(.data)
  }
  
  p <- NULL
  test <- FALSE
  if (!is.null(target)){
    
    if (!is.vector(target) && !is.matrix(target)){
      stop("The target must be either a vector or a matrix.")
    }
    
    if (!isSimplified(network)){
      if (!is.matrix(target))
        target <- as.matrix(target)
      p <- .predictARTMAP(network, .data, mTarget = target)
    } else{
      if (!is.vector(target)){
        stop("The simplified ARTMAP requires a vector for the target.")
      }
      p <- .predictARTMAP(network, .data, vTarget = target)
    }
  } else{
    p <- .predictARTMAP(network, .data)
    
  }
  return (p)
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
  classLabels <- as.character(classLabels)
  code <- .createDummyCodeMap(as.character(classLabels))
  return (code)
}

#' Label Encoding and Decoding
#' @description Convert numeric code to binary dummy code
#' @param classLabels A vector of class labels. They must be numeric.
#' @param dummyCode A list of dummy codes created for the labels using the function createDummyCodeMap
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
  labels <- .decode(dummyClasses, dummyCode)
  return (labels)
}

#' Data Normalization
#' @description Normalize each column of data to values between 0 and 1
#' @param .data A data object such as a data frame, a matrix , or an array.
#' @param use A data object that contains the same columns as those in .data. This dataset is
#' used to estimate the maximum and minimum values for normalizing .data. This is useful
#' when normalizing a test set using the same scales as the training set. If this data object
#' contains the attribute "ranges", then the minimum and maximum values from the "ranges"
#' will be used for the normalization. Finally, if use is NULL, then the scales are estimated 
#' directly from .data.
#' @return A normalized data object
#' @export
normalize <- function(.data, use = NULL){
  if (!is.null(use)){
    if (any(!colnames(use) %in% colnames(.data))){
      stop("Some of the columns in the .data object are not found in the use data object. Please make sure both of them contain 
           the same columns.")
    }
  }
  
  ranges <- array(dim = c(2, ncol(.data)))
  rownames(ranges) <- c("min", "max")
  for (n in seq_len(ncol(.data))){
    cmax <- cmin <- NA
    if (!is.null(use)){
      r <- attr(use, "ranges")
      if (!is.null(r)){
        cmin <- r[1,n]
        cmax <- r[2,n]
      } else{
        cmax <- max(use[, n], na.rm = T)
        cmin <- min(use[, n], na.rm = T)
      }
    } else{
      cmax <- max(.data[, n], na.rm = T)
      cmin <- min(.data[, n], na.rm = T)
    }
    .data[, n] <- (.data[, n] - cmin)/(cmax - cmin)
    ranges[, n] <- c(cmin, cmax)
  }
  attr(.data, "ranges") <- ranges
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

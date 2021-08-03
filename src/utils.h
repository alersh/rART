/****************************************************************************
 *
 *  utils.h
 *  ART utilities
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;

// sortIndex: Sort the values in descending order and return the indices of the values
NumericVector sortIndex( NumericVector x );

// colMax: get maximum values in all columns
NumericVector colMax( NumericMatrix x );

// colMin: get minimum values in all columns
NumericVector colMin( NumericMatrix x );

// sameCode: A function that returns the code x as is.
NumericVector sameCode( NumericVector x );

// complementCode: Create complement code
NumericVector complementCode( NumericVector x );

// uncomplementCode: Remove the complement part of the code and return the original
NumericVector uncomplementCode( NumericVector x );

// appendRows: append rows to a matrix
NumericMatrix appendRows( NumericMatrix x, int numRows );

// appendVector: append a vector
NumericVector appendVector( NumericVector v1, NumericVector v2 );

// initComplementWeight: Initialize weight and its complement
void initWeight( List net, bool complement );

NumericMatrix subsetRows( NumericMatrix x, int rows );

// vectorToMatrix: Set a NumericVector as a NumericMatrix
NumericVector vectorToMatrix( NumericVector v, int nrow, int ncol );

// linkClusters: Linking Clusters (nodes that are linked together)
List linkClusters( IntegerVector edges, IntegerVector nodes );

// Convert all numeric labels to their dummy codes
NumericMatrix encodeNumericLabel( NumericVector labels, NumericMatrix code );

// Convert all string labels to their dummy codes
NumericMatrix encodeStringLabel( StringVector labels, NumericMatrix code );

void printVector( NumericVector v );

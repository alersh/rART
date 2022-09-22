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

// appendRows: append rows to a matrix
NumericMatrix appendRows( NumericMatrix x, int numRows );

// appendColumns: append columns to a matrix
NumericMatrix appendColumns( NumericMatrix x, int numCols );

// lengthenVector: append length to a vector
template <class T> T appendVector( T x, int length );

// appendVector: append a vector
template <class T> T joinVectors( T v1, T v2 );

NumericMatrix subsetRows( NumericMatrix x, int rows );

NumericMatrix subsetColumns( NumericMatrix x, int cols );

IntegerVector subsetVector( IntegerVector x, int length );

// vectorToMatrix: Set a NumericVector as a NumericMatrix
NumericVector vectorToMatrix( NumericVector v, int nrow, int ncol );

// linkClusters: Linking Clusters (nodes that are linked together)
List linkClusters( IntegerVector edges, IntegerVector nodes );

// createDummyCodeMap: create dummy code for the class labels
List createDummyCodeMap ( StringVector classLabels );

// Convert all numeric labels to their dummy codes
NumericMatrix encodeNumericLabel( IntegerVector labels, List code );

// Convert all string labels to their dummy codes
NumericMatrix encodeStringLabel( StringVector labels, List code );

// convert the dummy codes to the original labels
StringVector decode (NumericMatrix dummyClasses, List dummyCode);

// check if two vectors are equal
bool equal( NumericVector x, NumericVector y );

void printVector( NumericVector v );

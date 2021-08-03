/****************************************************************************
 *
 *  fuzzy.h
 *  Fuzzy ART and Fuzzy ARTMAP
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;

bool isFuzzy ( List net );

namespace Fuzzy {
  double activation( List module, NumericVector x, NumericVector w );
  double TopoPredictActivation ( List module, NumericVector x, NumericVector w );
  double match( List module, NumericVector x, NumericVector w );
  NumericVector weightUpdate( List module, NumericVector x, NumericVector w );
  void trainART( List net, NumericMatrix x );
  void trainARTMAP( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue, Nullable< NumericMatrix > dummyLabels = R_NilValue );
  List predictART( List net, int id, NumericMatrix x );
  List predictARTMAP( List net, NumericMatrix x, Nullable< NumericVector > labels, Nullable< NumericMatrix > dummyLabels = R_NilValue, bool test = false );
}

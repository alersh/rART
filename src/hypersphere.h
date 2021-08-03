/****************************************************************************
 *
 *  hypersphere.cpp
 *  Hypersphere ART and Hypersphere ARTMAP
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;

bool isHypersphere ( List net );

namespace Hypersphere {

  void trainART( List net, NumericMatrix x );

  void trainARTMAP( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue, Nullable< NumericMatrix > dummyLabels = R_NilValue );

  List predictART( List net, int id, NumericMatrix x );

  List predictARTMAP( List net, NumericMatrix x, Nullable< NumericVector > labels, Nullable< NumericMatrix > dummyLabels = R_NilValue, bool test = false );

}

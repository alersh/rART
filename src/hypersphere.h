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

  void trainARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue );

  List predictART( List net, int id, NumericMatrix x );

  List predictARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false );

}

/****************************************************************************
 *
 *  art1.h
 *  ART 1
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;

#ifndef ART1_H
#define ART1_H

bool isART1 ( List net );
void createART1( List net, double L );

struct ART1 : IModel {
  
  ART1( List net );
  int getWeightDimension( int featureDimension );
  NumericVector newWeight( List module, NumericVector w );
  NumericVector getWbu( NumericVector w );
  NumericVector getWtd( NumericVector w );
  NumericVector intersect( NumericVector x, NumericVector w );
  NumericVector updateWtd( NumericVector x, NumericVector w_bu );
  NumericVector updateWbu( List module, NumericVector w_td );
  double activation( List module, NumericVector x, NumericVector w );
  double match(List module, NumericVector x, NumericVector w );
  NumericVector weightUpdate( List module, double learningRate, NumericVector x, NumericVector w );
  NumericVector processCode( NumericVector x );
  NumericVector unProcessCode( NumericVector x );
  NumericMatrix normalizeCode( NumericMatrix x );
};

#endif
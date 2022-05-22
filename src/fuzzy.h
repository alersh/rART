/****************************************************************************
 *
 *  fuzzy.h
 *  Fuzzy ART and Fuzzy ARTMAP
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;

#ifndef FUZZY_H
#define FUZZY_H

bool isFuzzy ( List net );

struct Fuzzy : IModel {
  
  Fuzzy( List net );
  int getWeightDimension( int featureDimension );
  NumericVector newWeight( NumericVector x );
  double activation( List module, NumericVector x, NumericVector w );
  double TopoPredictActivation (List module, NumericVector x, NumericVector w );
  double match(List module, NumericVector x, NumericVector w );
  NumericVector weightUpdate( List module, double learningRate, NumericVector x, NumericVector w );
  NumericVector processCode( NumericVector x );
  NumericVector unProcessCode( NumericVector x );
  NumericMatrix normalizeCode( NumericMatrix x );
};

#endif
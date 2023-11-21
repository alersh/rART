/****************************************************************************
 *
 *  hypersphere.cpp
 *  Hypersphere ART and Hypersphere ARTMAP
 *
 *
 *
 *  Reference:
 *  1. Anagnostopoulos, GC, Georgiopoulos, M. (2000) "Hypersphere ART and ARTMAP
 *  Unsupervised and Supervised, Incremental Learning", Proceedings of the
 *  IEEE-INNS-ENNS International Joint Conference on Neural Networks. Neural
 *  Computing: New Challenges and Perspectives for the New Millennium, 6.
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;

#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

bool isHypersphere ( List net );
void checkHypersphereBounds ( List net );

struct Hypersphere : IModel{
  Hypersphere( List net );
  Hypersphere( List net, NumericMatrix x );
  void initR_bar( NumericMatrix x );
  double norm( NumericVector x, NumericVector m );
  double R_bar( NumericMatrix x );
  int getWeightDimension( int featureDimension );
  NumericVector newWeight( List module, NumericVector w );
  double activation( List module, NumericVector x, NumericVector w );
  double TopoPredictActivation ( List module, NumericVector x, NumericVector w );
  double match( List module, NumericVector x, NumericVector w );
  NumericVector weightUpdate( List module, double learningRate, NumericVector x, NumericVector w );
  NumericVector getNextLayerInput( NumericVector w );
  NumericVector processCode( NumericVector x );
  NumericVector unProcessCode( NumericVector x );
  
};

#endif
/****************************************************************************
 *
 *  fuzzy.cpp
 *  Fuzzy ART and Fuzzy ARTMAP
 *
 *
 *
 *  References:
 *  1. Carpenter, GA, Grossberg, S, Rosen DB. (1991) "Fuzzy ART: Fast stable
 *  learning and categorization of analog patterns by an adaptive resonance
 *  system", Neural Networks, 4(6), pp 759-771.
 *  2. Carpenter, GA, Grossberg, S, Markuzon, N, Reynolds, JH, Rosen, DB. (1992)
 *  "Fuzzy ARTMAP: A neural network architecture for incremental supervised
 *  learning of analog multidimensional maps", IEEE Transactions on Neural Networks,
 *  3(5), pp. 698-713.
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "ART.h"
#include "utils.h"
using namespace Rcpp;
#include "fuzzy.h"

bool isFuzzy ( List net ){
  return as<std::string>( net.attr( "rule" ) ).compare( "fuzzy" ) == 0;
}

Fuzzy::Fuzzy( List net ) : IModel( net ){}

int Fuzzy::getWeightDimension( int featureDimension ){
  return featureDimension * 2;
}

NumericVector Fuzzy::newWeight( NumericVector w ){
  return w;
}

double Fuzzy::activation( List module, NumericVector x, NumericVector w ) {
  
  return sum( na_omit( pmin( x, w ) ) )/( ART::getAlpha( module ) + sum( na_omit( w ) ) );
}

double Fuzzy::TopoPredictActivation ( List module, NumericVector x, NumericVector w ) {
  
  double a = 1 - sum( na_omit( pmin( x, w ) - w ) )/ART::getDimension( module );
  return a;
}

double Fuzzy::match( List module, NumericVector x, NumericVector w )  {
  
  return sum( na_omit( pmin( x, w ) ) )/sum( na_omit( x ) );
}

NumericVector Fuzzy::weightUpdate( List module, double learningRate, NumericVector x, NumericVector w ) {
  return learningRate * pmin( x, w ) + ( 1.0 - learningRate ) * w;
  
}

// processCode: Create complement code
NumericVector Fuzzy::processCode( NumericVector x )  {
  int s = x.size();
  NumericVector c( s*2 );
  for ( int i = 0; i < s; i++ ){
    c[i] = x[i];
    c[i+s] = 1 - x[i];
  }
  return c;
}

NumericVector Fuzzy::unProcessCode( NumericVector x ){
  int l = x.length();
  // the length must be even
  if ( l % 2 != 0 )
    stop( "The length of the code must be an even number." );
  return x[ Range( 0, l/2 - 1 ) ];
}


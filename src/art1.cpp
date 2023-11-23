/****************************************************************************
 *
 *  art1.cpp
 *  ART 1
 *
 *
 *
 *  References:
 *  1. da Silva, L.E.B, Elnabarawy, I., and Wunsch II, D.C. (2019) "A survey of 
 *  Adaptive Resonance Theory neural network models for engineering applications", 
 *  Neural Networks, 120, pp 167 - 203.
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "ART.h"
#include "ARTMAP.h"
#include "utils.h"
using namespace Rcpp;
#include "art1.h"

bool isART1 ( List net ){
  return as<std::string>( net.attr( "rule" ) ).compare( "ART1" ) == 0;
}

// [[Rcpp::export(.checkART1Bounds)]]
void checkART1Bounds ( List net ){
  
  int numModules = ART::getNumModules( net );
  for ( int i = 0; i < numModules; i++ ){
    List module = ART::getModule( net, i );
    double learningRate = ART::getLearningRate( module );
    if ( learningRate < 0.0 ){
      stop( "The learningRate value must be larger than 0." );
    }
  }
}

ART1::ART1( List net ) : IModel( net ){}

int ART1::getWeightDimension( int featureDimension ){
  return featureDimension * 2;
}

NumericVector ART1::newWeight( List module, NumericVector w ){
  NumericVector w_td = rep( 1.0, w.length() );
  NumericVector w_bu = updateWbu( module, w_td );
  NumericVector w_new = joinVectors( w_bu, w_td );
  return w_new;
}

NumericVector ART1::getWbu( NumericVector w ){
  int length = w.length()/2;
  return w[Range( 0, length - 1 )];
}

NumericVector ART1::getWtd( NumericVector w ){
  int half = w.length()/2;
  return w[Range( half, w.length() - 1 )];
}

double ART1::activation( List module, NumericVector x, NumericVector w ) {
  NumericVector w_bu = getWbu( w );
  int dim = w_bu.length();
  double T = 0;
  for ( int i = 0; i < dim; i++ ){
    if ( !NumericVector::is_na( x[i] ) ){
      T += x[i] * w_bu[i];
    }
  }
  
  return T;
}

NumericVector ART1::intersect( NumericVector x, NumericVector w ){
  NumericVector intersect =  x * w;
  return intersect;
}

NumericVector ART1::updateWtd( NumericVector x, NumericVector w_bu ){
  return intersect( x, w_bu );
}

NumericVector ART1::updateWbu( List module, NumericVector w_td ){
  //double L = as<double>( module["L"] );
  double L = ART::getLearningRate( module );
  return L/( L - 1 + sum( w_td ) ) * w_td;
}

double ART1::match( List module, NumericVector x, NumericVector w )  {
  int dim = w.length();
  NumericVector w_td = getWtd( w );
  double match = sum( na_omit( intersect( x, w_td ) ) )/sum( na_omit( x ) );
  return match;
}

NumericVector ART1::weightUpdate( List module, double learningRate, NumericVector x, NumericVector w ) {
  int dim = ART::getWeightDimension( module );
  NumericVector w_td = getWtd( w );
  NumericVector w_td_new = intersect( x, w_td );
  //double L = as<double>( module["L"] );
  double L = ART::getLearningRate( module );
  NumericVector w_bu_new = updateWbu( module, w_td_new );
  
  NumericVector w_new = joinVectors( w_bu_new, w_td_new );
  
  return w_new;
}

NumericVector ART1::getNextLayerInput( NumericVector w ){
  return w;
}

NumericVector ART1::processCode( NumericVector x )  {
  return x;
}

NumericVector ART1::unProcessCode( NumericVector x ){
  return x;
}


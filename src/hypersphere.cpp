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
#include "ART.h"
//#include "ARTMAP.h"
#include "utils.h"
#include "hypersphere.h"
using namespace Rcpp;


bool isHypersphere ( List net ){
  return as<std::string>( net.attr( "rule" ) ).compare( "hypersphere" ) == 0;
}

Hypersphere::Hypersphere( List net ) : IModel( net ){}

Hypersphere::Hypersphere( List net, NumericMatrix x ) : IModel( net ){
  
  // add R_bar to each module
  initR_bar( x );
}

void Hypersphere::initR_bar( NumericMatrix x ){
  double rbar = R_bar( x );
  int numModules = ART::getNumModules( this->net );
  for ( int i = 0; i < numModules; i++ ){
    List module = ART::getModule( this->net, i );
    module.push_back( rbar, "R_bar" );
    ART::setModule( this->net, module );
  }
}

double Hypersphere::norm( NumericVector x, NumericVector m ){
  return sqrt( sum( na_omit( pow( x - m, 2 ) ) ) );
}

double Hypersphere::R_bar( NumericMatrix x ){
  NumericVector maximum = colMax( x );
  NumericVector minimum = colMin( x );
  
  double d = sqrt( 0.5 * sum( na_omit( pow( maximum - minimum, 2 ) ) ) );
  return d;
}

int Hypersphere::getWeightDimension( int featureDimension ){
  return featureDimension + 1;
}

NumericVector Hypersphere::newWeight( NumericVector w ){
  /* add the radius element to the new weight vector */
  w.push_back(0);
  return w;
}

double Hypersphere::activation( List module, NumericVector x, NumericVector w ){
  int dimension = w.length();
  double R = w[dimension - 1];
  NumericVector m = w[Range( 0, dimension - 2 )];
  double R_bar = module["R_bar"];
  double maximum = max( na_omit( NumericVector::create( R, norm( x, m ) ) ) );
  double a = ( R_bar - maximum )/( R_bar - R + ART::getAlpha( module ) );
  
  return a;
}

double Hypersphere::TopoPredictActivation ( List module, NumericVector x, NumericVector w ){
  int dimension = w.length();
  double R = w[dimension - 1];
  NumericVector m = w[Range( 0, dimension - 2 )];
  double R_bar = module["R_bar"];
  double maximum = max( na_omit( NumericVector::create( norm( x, m ) - R, 0 ) ) );
  return 1 - maximum/( 2*R_bar );
}

double Hypersphere::match( List module, NumericVector x, NumericVector w ){
  int dimension = w.length();
  double R = w[dimension - 1];
  NumericVector m = w[Range( 0, dimension - 2 )];
  double R_bar = module["R_bar"];
  double maximum = max( na_omit( NumericVector::create( R, norm( x, m ) ) ) );
  return 1 - maximum/R_bar;
}

NumericVector Hypersphere::weightUpdate( List module, double learningRate, NumericVector x, NumericVector w ){
  
  int dimension = w.length();
  double R = w[dimension - 1];
  NumericVector m = w[Range( 0, dimension - 2 )];
  double dis = norm( x, m );
  NumericVector mnew;
  if ( dis < 0.000001 ){
    mnew = m;
  }
  else{
    double minimum = min( na_omit( NumericVector::create( R, dis ) ) );
    mnew = m + learningRate/2 * ( x - m ) * ( 1 - minimum/dis );
  }
  double Rnew;
  double maximum = max( na_omit( NumericVector::create( R, dis ) ) );
  Rnew = R + learningRate/2 * ( maximum - R );
  mnew.push_back( Rnew );
  return mnew;
}

NumericVector Hypersphere::processCode( NumericVector x ) {
  return x;
}

NumericVector Hypersphere::unProcessCode( NumericVector x ){
  return x;
}
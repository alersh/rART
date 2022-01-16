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
#include "ARTMAP.h"
#include "TopoART.h"
#include "utils.h"
#include "fuzzy.h"
using namespace Rcpp;
using namespace std;


bool isHypersphere ( List net ){
  return as<string>( net.attr( "rule" ) ).compare( "hypersphere" ) == 0;
}

namespace Hypersphere {

  double norm( NumericVector x, NumericVector m ){
    return sqrt( sum( na_omit( pow( x - m, 2 ) ) ) );
  }

  double R_bar( NumericMatrix x ){
    NumericVector maximum = colMax( x );
    NumericVector minimum = colMin( x );

    double d = sqrt( 0.5 * sum( na_omit( pow( maximum - minimum, 2 ) ) ) );
    return d;
  }

  void initWeight( List net, NumericMatrix x ){
    int numModules = net["numModules"];
    double rbar = R_bar( x );
    for ( int i = 0; i < numModules; i++ ){
      List module = as<List>( net["module"] )[i];
      module.push_back( rbar, "R_bar" );
      NumericMatrix w = no_init( module["capacity"], as<int>( module["dimension"] ) + 1 );
      module["w"] = w;
      as<List>( net["module"] )( i ) = module;
    }

    if ( isARTMAP( net ) ){
      List module_ab = as<List>( net["mapfield"] )["ab"];
      module_ab.push_back( rbar, "R_bar" );
      as<List>( net["mapfield"] )["ab"] = module_ab;
    }
  }

  double activation( List module, NumericVector x, NumericVector w ){
    int dimension = module["dimension"];
    double R = w[dimension];
    NumericVector m = w[Range( 0, dimension - 1 )];
    double R_bar = module["R_bar"];
    double maximum = max( na_omit( NumericVector::create( R, norm( x, m ) ) ) );
    double a = ( R_bar - maximum )/( R_bar - R + as<double>( module["alpha"] ) );

    return a;
  }

  double TopoPredictActivation ( List module, NumericVector x, NumericVector w ){
    int dimension = module["dimension"];
    double R = w[dimension];
    NumericVector m = w[Range( 0, dimension - 1 )];
    double R_bar = module["R_bar"];
    double maximum = max( na_omit( NumericVector::create( norm( x, m ) - R, 0 ) ) );
    return 1 - maximum/( 2*R_bar );
  }

  double match( List module, NumericVector x, NumericVector w ){
    int dimension = module["dimension"];
    double R = w[dimension];
    NumericVector m = w[Range( 0, dimension - 1 )];
    double R_bar = module["R_bar"];
    double maximum = max( na_omit( NumericVector::create( R, norm( x, m ) ) ) );
    return 1 - maximum/R_bar;
  }

  NumericVector weightUpdate( List module, NumericVector x, NumericVector w ){

    int dimension = module["dimension"];
    double b = module["beta"];
    double R = w[dimension];
    NumericVector m = w[Range( 0, dimension - 1 )];
    double dis = norm( x, m );
    NumericVector mnew;
    if ( dis < 0.000001 ){
      mnew = m;
    }
    else{
      double minimum = min( na_omit( NumericVector::create( R, dis ) ) );
      mnew = m + b/2 * ( x - m ) * ( 1 - minimum/dis );
    }
    double Rnew;
    double maximum = max( na_omit( NumericVector::create( R, dis ) ) );
    Rnew = R + b/2 * ( maximum - R );
    mnew.push_back( Rnew );
    return mnew;
  }

  void newCategory( List module, NumericVector x ){

    int n = 0;
    if ( as<int>( module["numCategories"] ) > 0 ){
      if (as<int>( module["numCategories"] ) == as<NumericMatrix>( module["w"] ).rows() ){
        // reached the max capacity, so add more rows
        module["w"] = appendRows( module["w"], module["capacity"] );
      }
      int n = as<int>( module["numCategories"] );
      module["numCategories"] = n + 1;
    }

    NumericVector w = x;
    w.push_back( 0 ); // R
    as<NumericMatrix>( module["w"] )( n,_ ) = w;

  }

  void trainART( List net, NumericMatrix x ){
    Hypersphere::initWeight( net, x );
    if ( isART( net ) ){
      ART::train( net, x, sameCode, activation, match, weightUpdate );
    }
    else if ( isTopoART( net ) ){
      Topo::train( net, x, sameCode, activation, match, weightUpdate);
    }

  }

  void trainARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue ){
    if ( !isSimplified( net ) ){
      stop( "Only the simplified form is available for hypersphere learning." );
    }

    Hypersphere::initWeight( net, x );
    if ( isARTMAP( net ) ){
      ARTMAP::train( net, x, vTarget, mTarget, sameCode, activation, match, weightUpdate, weightUpdate );
    }

  }

  List predictART( List net, int id, NumericMatrix x ){
    List results;
    if ( isART( net ) ){
      results = ART::predict ( net, id, x, sameCode, activation, match );
    }
    else if ( isTopoART( net ) ){
      results = Topo::predict( net, id, x, sameCode, TopoPredictActivation, match );
    }
    return results;
  }

  List predictARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false ){
    List results;
    if ( isARTMAP( net ) ){
      if ( !isSimplified( net ) ){
        // only the simplified form is available
        net.attr( "simplified" ) = true;
      }
      results = ARTMAP::predict ( net, x, vTarget, mTarget, sameCode, sameCode, activation, match, test );
    }

    return results;
  }

}


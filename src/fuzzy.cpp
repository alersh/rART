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
#include "ARTMAP.h"
#include "TopoART.h"
#include "utils.h"
using namespace Rcpp;
using namespace std;


bool isFuzzy ( List net ){
  return as<string>( net.attr( "rule" ) ).compare( "fuzzy" ) == 0;
}

namespace Fuzzy {

  double activation( List module, NumericVector x, NumericVector w ){
    return sum( na_omit( pmin( x, w ) ) )/( as<double>( module["alpha"] ) + sum( na_omit( w ) ) );
  }

  double TopoPredictActivation ( List module, NumericVector x, NumericVector w ){
    double a = 1 - sum( na_omit( pmin( x, w ) - w ) )/as<int>( module["dimension"] );
    return a;
  }

  double match( List module, NumericVector x, NumericVector w ){
    return sum( na_omit( pmin( x, w ) ) )/sum( na_omit( x ) );
  }

  NumericVector weightUpdate( List module, NumericVector x, NumericVector w ){
    double b = as<double>( module["beta"] );

    NumericVector p = b * pmin( x, w ) + ( 1.0 - b ) * w;
    return p;
  }

  void trainART( List net, NumericMatrix x ){
    int numModules = as<int>( net["numModules"] );
    for ( int i = 0; i < numModules; i++ ){
      initWeight( as<List>( net["module"] )[i], true );
    }
    if ( isART( net ) ){
      ART::train( net, x, complementCode, activation, match, weightUpdate );
    }
    else if ( isTopoART( net ) ){
      Topo::train( net, x, complementCode, activation, match, weightUpdate );
    }

  }

  void trainARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue ){
    int numModules = as<int>( net["numModules"] );
    for ( int i = 0; i < numModules; i++ ){
      initWeight( as<List>( net["module"] )[i], true );
    }
    if ( isARTMAP( net ) ){
      ARTMAP::train( net, x, vTarget, mTarget, complementCode, activation, match, weightUpdate, weightUpdate );
    }

  }

  List predictART( List net, int id, NumericMatrix x ){
    List results;
    if ( isART( net ) ){
      results = ART::predict ( net, id, x, complementCode, activation, match );
    }
    else if ( isTopoART( net ) ){
      results = Topo::predict( net, id, x, complementCode, TopoPredictActivation, match );
    }
    return results;
  }

  List predictARTMAP( List net, NumericMatrix x, Nullable< NumericVector > vTarget, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false ){
    List results;
    if ( isARTMAP( net ) ){
      results = ARTMAP::predict ( net, x, vTarget, mTarget, complementCode, uncomplementCode, activation, match, test );
    }
    return results;
  }

}



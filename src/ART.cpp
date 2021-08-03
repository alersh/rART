/****************************************************************************
 *
 *  ART.cpp
 *  ART
 *
 ****************************************************************************/


#include <Rcpp.h>
#include "utils.h"
#include "fuzzy.h"
#include "hypersphere.h"
using namespace Rcpp;
using namespace std;


namespace ART {

  List module ( int id, int numFeatures, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100 ){
    NumericMatrix w;
    NumericVector stm;
    List module = List::create( _["id"] = id,                   // module id
                                _["capacity"] = categorySize,   // number of category to create when the module runs out of categories to match
                                _["numCategories"] = 0,         // number of categories created during learning
                                _["numFeatures"] = numFeatures, // number of features/dimensions in the data
                                _["alpha"] = 0.001,             // activation function parameter
                                _["epsilon"] = 0.000001,        // match function parameter
                                _["w"] = w,                     // top-down weights
                                _["rho"] = vigilance,           // vigilance parameter
                                _["beta"] = learningRate,       // learning parameter
                                _["Jmax"] = -1                  // the node index with the highest activation and the best match
    );

    module.attr( "class" ) = "ART";
    return module;
  }

  NumericVector activation( List net, NumericVector x, std::function< double( List, NumericVector, NumericVector ) > fun ){
    int nc = net["numCategories"];
    NumericVector a( nc );

    for ( int k = 0; k < nc; k++ ){
      NumericVector w = as<NumericMatrix>( net["w"] )( k, _ );
      a[k] = fun( net, x, w );
    }

    return a;
  }

  double match( List net, NumericVector x, int weightIndex, std::function< double( List, NumericVector, NumericVector ) > fun ){
    NumericVector w = as<NumericMatrix>( net["w"] )( weightIndex, _ );
    double a = fun( net, x, w );
    return a;
  }


  int weightUpdate( List net, NumericVector x, int weightIndex, std::function< NumericVector( List, NumericVector, NumericVector ) > fun ){
    NumericMatrix w_old = as<NumericMatrix>( net["w"] );
    NumericMatrix wm = as<NumericMatrix>( net["w"] );
    NumericVector w =  as<NumericMatrix>( wm )( weightIndex, _ );

    wm( weightIndex, _ ) = fun( net, x, w );
    int change = 0;
    double s = sum( abs ( w_old( weightIndex, _ ) - wm( weightIndex, _ ) ));
    if ( s > 0.0000001 ){
      change++;
    }
    return change;
  }

  void newCategory( List net, NumericVector x ){

    if ( as<int>( net["numCategories"] ) == 0 ){
      as<NumericMatrix>( net["w"] )( 0,_ ) = x;
      net["numCategories"] = as<int>( net["numCategories"] ) + 1;
    }
    else{
      if (as<int>( net["numCategories"] ) == as<NumericMatrix>( net["w"] ).rows() ){
        // reached the max capacity, so add more rows
        NumericMatrix wm = as<NumericMatrix>( net["w"] );
        int nrow = wm.nrow();
        int ncol = wm.ncol();
        NumericMatrix newWeight ( nrow + as<int>( net["capacity"] ), ncol );
        for (int i = 0; i < nrow; i++){
          newWeight ( i,_ ) = wm( i,_ );

        }
        net["w"] = newWeight;
      }
      int n = as<int>( net["numCategories"] ) + 1;
      net["numCategories"] = n;
      as<NumericMatrix>( net["w"] )( n-1,_ ) = x;

    }

  }

  int learn( List net,
             int id,
             NumericVector d,
             std::function< double( List, NumericVector, NumericVector ) > activationFun,
             std::function< double( List, NumericVector, NumericVector ) > matchFun,
             std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun ){
    List module = as<List>( net["module"] )[id];
    int change = 0;

    int nc = module["numCategories"];
    if ( nc == 0 ){
      newCategory( module, d );
      change++;
    }
    else{
      NumericVector a = activation( module, d, activationFun );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      while( !resonance ){
        int J_max = T_j( j );
        double m = match( module, d, J_max, matchFun );
        if ( m >= as<double>( module["rho"] ) ){
          module["Jmax"] = J_max;
          change += weightUpdate( module, d, J_max, weightUpdateFun );

          resonance = true;
        }
        else{
          if ( j == as<int>( module["numCategories"] ) - 1 ){
            module["Jmax"] = j + 1;
            newCategory( module, d );
            change++;
            resonance = true;
          }
          else{
            j++;
          } // if
        } // match
      } // while resonance
    } // if

    return change;
  }

  int classify ( List net,
                 int id,
                 NumericVector d,
                 std::function< double( List, NumericVector, NumericVector ) > activationFun,
                 std::function< double( List, NumericVector, NumericVector ) > matchFun ){
    List module = as<List>( net["module"] )[id];
    int category = -1;

    NumericVector a = activation( module, d, activationFun );
    NumericVector T_j = sortIndex( a );
    bool resonance = false;
    int j = 0;

    while(!resonance){
      int J_max = T_j( j );
      double m = match( module, d, J_max, matchFun );
      if ( m >= as<double>( module["rho"] ) ){
        module["Jmax"] = J_max;
        category = J_max;
        resonance = true;
      }
      else{
        if ( j  == as<int>( module["numCategories"] ) - 1 ){
          module["Jmax"] = category;
          resonance = true;
        }
        else{
          j++;
        } // if
      } // match
    } // while resonance

    return category;
  }

  void train( List net,
              NumericMatrix x,
              std::function< NumericVector( NumericVector ) > codeFun,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun ){

    int ep = as<int>( net["maxEpochs"] );
    int nrow = x.rows();
    for (int i = 1; i <= ep; i++){
      int change = 0;
      std::cout << "Epoch no. " << i << std::endl;

      // currently, only one module is supported
      int id = as<List>( as<List>( net["module"] )[0] )["id"];
      for (int i = 0; i < nrow; i++){
        change += learn( net, id, codeFun( x( i, _ ) ), activationFun, matchFun, weightUpdateFun );
      }

      std::cout << "Number of changes: " << change << std::endl;
      if ( change == 0) {
        net["epochs"] = i;
        break;
      }
    }
    // subset weight matrix
    int l = net["numModules"];
    for ( int i = 0; i < l; i++ ){
      List module = as<List>( net["module"] )[i];
      module["w"] = subsetRows( module["w"], module["numCategories"] );
    }
  }

  List predict( List net,
                int id,
                NumericMatrix x,
                std::function< NumericVector( NumericVector ) > codeFun,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun ){
    List classified;
    int nrow = x.rows();
    NumericVector category( nrow );

    for (int i = 0; i < nrow; i++){
        // currently supports only one module
        int result = classify( net["module"], id, codeFun( x( i,_ ) ), activationFun, matchFun );
        if ( result == -1 ){
          category( i ) = NA_INTEGER;
        }
        else{
          category( i ) = result;
        }

    }
    classified = List::create( _["category"] = category );

    return classified;

  }

}

// [[Rcpp::export(.trainART)]]
void trainART ( List net, NumericMatrix x ){

  if ( isFuzzy( net ) ){
    Fuzzy::trainART( net, x );
  }
  if ( isHypersphere( net ) ){
    Hypersphere::trainART( net, x );
  }

}


// [[Rcpp::export(.predictART)]]
List predictART ( List net, int id, NumericMatrix x ){

  List results;
  if ( isFuzzy( net ) ){
    results = Fuzzy::predictART( net, id, x );
  }
  if ( isHypersphere( net ) ){
    results = Hypersphere::predictART( net, id, x );
  }

  return results;
}

// [[Rcpp::export(.ART)]]
List newART ( int numFeatures, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20 ){
  List net = List::create( _["numModules"] = num,       // number of ART modules
                           _["epochs"] = 0,             // total number of epochs required to learn
                           _["maxEpochs"] = maxEpochs   // maximum number of epochs
  );

  List modules;
  for ( int i = 0; i < num; i++ ){
    modules.push_back( ART::module( i, numFeatures, vigilance, learningRate, categorySize ) );
  }

  net.push_back( modules, "module" );

  net.attr( "class" ) = CharacterVector::create( "ART" );
  return net;

}

bool isART ( List net ){
  return as<string>( net.attr( "class" ) ).compare( "ART" ) == 0;
}


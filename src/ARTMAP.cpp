/****************************************************************************
 *
 *  ARTMAP.cpp
 *  ARTMAP
 *
 ****************************************************************************/


#include <Rcpp.h>
#include "ART.h"
#include "utils.h"
#include "fuzzy.h"
#include "hypersphere.h"
using namespace Rcpp;
using namespace std;


bool isARTMAP ( List net ){
  return as<string>( net.attr( "class" ) ).compare( "ARTMAP" ) == 0;
}

bool isSimplified( List net ){
  return as<bool>( net.attr( "simplified" ) );
}

namespace ARTMAP {
  // create and return a mapfield module
  List mapfield ( int id, int numFeatures, double vigilance = 0.75, double learningRate = 1.0, bool simplified = false ){
    List module;
    NumericMatrix w;
    if ( !simplified ){
      module = List::create( _["id"] = id,                   // number of category to create when the module runs out of categories to match
                             _["numFeatures"] = numFeatures, // number of features
                             _["a_size"] = 0,                // number of categories from ART a
                             _["b_size"] = 0,                // number of categories from ART b
                             _["epsilon"] = 0.000001,        // match function parameter
                             _["w"] = w,                     // map field weights
                             _["rho"] = vigilance,           // vigilance parameter
                             _["beta"] = learningRate,       // learning parameter
                             _["Jmax"] = -1                  // the node index with the highest activation and the best match
      );
    }
    else{
      module = List::create( _["id"] = id,                   // number of category to create when the module runs out of categories to match
                             _["numFeatures"] = numFeatures, // number of features
                             _["numMapfield"] = 0,           // number of categories in the mapfield
                             _["w"] = w,                     // map field weights
                             _["rho"] = vigilance,           // vigilance parameter
                             _["beta"] = learningRate        // learning parameter
      );
    }
    return module;
  }

  namespace simplified {

    void newCategory( List module_ab, int label ){

      IntegerVector m = as<IntegerVector>( module_ab["w"] );
      m.push_back( label );
      module_ab["w"] = m;
      module_ab["numMapfield"] = as<int>( module_ab["numMapfield"] ) + 1;

    }

    int learn ( List net,
                NumericVector d,
                int label,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun ){
      List module = as<List>( net["module"] )["a"];
      List module_ab = as<List>( net["mapfield"] )["ab"];

      int change = 0;

      int nc = module["numCategories"];
      if ( nc == 0 ){

        ART::newCategory( module, d );
        newCategory( module_ab, label );
        change++;

      }
      else{
        NumericVector a = ART::activation( module, d, activationFun );
        NumericVector T_j = sortIndex( a );
        bool resonance = false;
        int j = 0;
        double rho = module["rho"];
        while( !resonance ){
          int J_max = T_j( j );

          double m = ART::match( module, d, J_max, matchFun );
          NumericMatrix w_a = module["w"];

          if ( m >= rho ){

            NumericVector mapfield = module_ab["w"];

            if ( mapfield[J_max] == label ){
              module["Jmax"] = J_max;
              change += ART::weightUpdate( module, d, J_max, weightUpdateFun );
              resonance = true;
            }
            else{

              rho = std::min( m + as<double>( module["epsilon"] ), 1.0 );

              if ( j == as<int>( module["numCategories"] ) - 1 ){
                module["Jmax"] = j + 1;
                ART::newCategory( module, d );

                newCategory( module_ab, label );

                change++;
                resonance = true;
              }
              else{
                j++;
              } // if
            } // mapfield == label
          } // match >= rho_a
          else{
            if ( j == as<int>( module["numCategories"] ) - 1 ){
              module["Jmax"] = j + 1;
              ART::newCategory( module, d );

              newCategory( module_ab, label );
              change++;
              resonance = true;
            }
            else{
              j++;
            } // if
          } // match fails
        } // while resonance
      } // if

      return change;
    }

    int test( int predicted, int label ){
      int matched = NA_INTEGER;

      if ( predicted == label ){
        matched = 1;
      }
      else{
        matched = 0;
      }
      return matched;
    }

    // simplified classification
    List classify( List net,
                   NumericVector d,
                   std::function< double( List, NumericVector, NumericVector ) > activationFun,
                   std::function< double( List, NumericVector, NumericVector ) > matchFun,
                   bool test ){

      List module = as<List>( net["module"] )["a"];
      List module_ab = as<List>( net["mapfield"] )["ab"];
      int category = NA_INTEGER;
      int predicted = NA_INTEGER;

      IntegerVector mapfield = module_ab["w"];
      int nc = as<int>( module["numCategories"] );

      NumericVector a = ART::activation( module, d, activationFun );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho = module["rho"];
      while( !resonance ){
        int J_max = T_j( j );

        double m = ART::match( module, d, J_max, matchFun );

        if ( m >= rho ){

          category = J_max;
          predicted = mapfield[J_max];
          resonance = true;

        } // match >= rho_a
        else{
          if ( j == nc - 1 ){
            // can't find a match
            module["Jmax"] = NA_INTEGER;
            resonance = true;
          }
          else{
            // search for the next best matching node in F2a
            j++;
          } // if
        } // match fails
      } // while resonance

      return ( List::create( _["category_a"] = category,
                             _["predicted"] = predicted ) );
    }
  }

  namespace standard {
    // recall reactivates the F2b node based on the mapfield weights to retrieve its F1b pattern
    NumericVector recall( List module_b, NumericVector mapfield ){
      // find which mapfield node is active (either 1 or 0)
      int l = mapfield.length();
      int nodeIndex_b = -1;
      for ( int i = 0; i < l; i++ ){
        if ( mapfield[i] == 1 ){
          nodeIndex_b = i;
          break;
        }
      }

      NumericVector F1;
      if ( nodeIndex_b == -1 ){
        F1( as<NumericMatrix>( module_b["w"] ).cols() );
        // Can't find the node b in F2. Something is wrong. Return all NA
        F1.fill( NA_INTEGER );
        return F1;
      }
      F1 = as<NumericMatrix>( module_b["w"] )( nodeIndex_b, _ );
      return F1;
    }

    double match( List module_ab, int nodeIndex_a, int nodeIndex_b, std::function< double( List, NumericVector, NumericVector ) > fun ){
      NumericMatrix w_ab = module_ab["w"];
      NumericVector w_a = w_ab( nodeIndex_a, _ );

      NumericVector yb( w_a.length(), 0 );

      yb( nodeIndex_b ) = 1;
      double a = fun( module_ab, yb, w_a );

      return a;
    }

    int mapfieldUpdate( List module_ab, int nodeIndex_a, int nodeIndex_b, std::function< NumericVector( List, NumericVector, NumericVector ) > fun ){
      int change = 0;
      NumericMatrix w_ab = module_ab["w"];
      NumericVector w_a = w_ab( nodeIndex_a, _ );

      // activate y_b
      NumericVector y_b( w_a.length(), 0 );
      y_b( nodeIndex_b ) = 1;
      // no new F2 node has been added in modules a and b
      w_ab( nodeIndex_a, _ ) = fun( module_ab, y_b, w_a );
      module_ab["w"] = w_ab;

      return change++;
    }

    void newCategory_a( List module_ab ){
      NumericMatrix w = module_ab["w"];
      int rows = w.rows();
      int cols = w.cols();
      NumericMatrix wnew( rows + 1, cols );
      // add a new column
      for ( int r = 0; r < rows; r++ ){
        for ( int c = 0; c < cols; c++ ){
          wnew( r,c ) = w( r,c );
        }
      }
      for ( int c = 0; c < cols; c++ ){
        // uncommitted weights are all set to 1
        wnew( rows, c ) = 1;

      }
      module_ab["w"] = wnew;
      module_ab["a_size"] = rows + 1;
    }

    void newCategory_b( List module_ab ){
      NumericMatrix w = module_ab["w"];
      int rows = w.rows();
      int cols = w.cols();
      NumericMatrix wnew( rows, cols + 1 );
      // add a new column
      for ( int r = 0; r < rows; r++ ){
        for ( int c = 0; c < cols; c++ ){
          wnew( r,c ) = w( r,c );
        }
        wnew( r, cols ) = 0;
      }
      // set the new node to 1
      wnew( rows - 1, cols ) = 0;
      module_ab["w"] = wnew;
      module_ab["b_size"] = cols + 1;
    }

    int learn ( List net,
                NumericVector d,
                NumericVector label,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > mapfieldUpdateFun ){
      int change = 0;
      List module_a = as<List>( net["module"] )["a"];
      List module_b = as<List>( net["module"] )["b"];
      List module_ab = as<List>( net["mapfield"] )["ab"];

      // init
      int nc_a = module_a["numCategories"];
      int nc_b = module_b["numCategories"];
      int nc_ab_a = module_ab["a_size"];
      int nc_ab_b = module_ab["b_size"];
      if ( nc_a == 0  && nc_b == 0  && nc_ab_a == 0 && nc_ab_b == 0 ){

        ART::newCategory( module_a, d );
        ART::newCategory( module_b, label );
        // Add new category in b first before a
        newCategory_b( module_ab );
        newCategory_a( module_ab );
        change += mapfieldUpdate( module_ab, 0, 0, mapfieldUpdateFun );

      }
      else{
        change += ART::learn( net, module_b["id"], label, activationFun, matchFun, weightUpdateFun );
        // add a new ab category whenever a new category is added in F2b
        if ( as<int>( module_b["numCategories"] ) > as<int>( module_ab["b_size"] ) ){
          // update nodes in mapfield
          newCategory_b( module_ab );

          change++;
        }
        // get ART a F2 activations
        NumericVector a = ART::activation( module_a, d, activationFun );
        NumericVector T_j = sortIndex( a );
        bool resonance = false;
        int j = 0;
        double rho_a = module_a["rho"];
        while( !resonance ){
          int Jmax_a = T_j( j );

          double m = ART::match( module_a, d, Jmax_a, matchFun );

          if ( m >= rho_a ){
            // check the mapfield

            double m_ab = match( module_ab, Jmax_a, module_b["Jmax"], matchFun );

            if ( m_ab >= as<double>( module_ab["rho"] ) ){
              module_a["Jmax"] = Jmax_a;
              change += ART::weightUpdate( module_a, d, Jmax_a, weightUpdateFun );
              change += mapfieldUpdate( module_ab, Jmax_a, module_b["Jmax"], mapfieldUpdateFun );

              resonance = true;
            }

            if ( !resonance ){

              rho_a = std::min( m + as<double>( module_a["epsilon"] ), 1.0 );

              // if run out of categories, then add a new one
              if ( j == as<int>( module_a["numCategories"] ) - 1 ){
                j++;
                module_a["Jmax"] = j;

                // add a new F2 node in ART a
                ART::newCategory( module_a, d );

                // add a new node in ART ab
                newCategory_a( module_ab );
                change += mapfieldUpdate( module_ab, j, module_b["Jmax"], mapfieldUpdateFun );

                resonance = true;
              }
              else{
                j++;
              } // if
            } // resonance
          }
          else{
            if ( j == as<int>( module_a["numCategories"] ) - 1 ){
              j++;
              module_a["Jmax"] = j;
              ART::newCategory( module_a, d );

              // add a new node in ART ab
              newCategory_a( module_ab );

              change += mapfieldUpdate( module_ab, j, module_b["Jmax"], mapfieldUpdateFun );

              resonance = true;

            }
            else{
              j++;
            } // if
          } // match fails
        } // while resonance
      } // if

      return change;
    }

    int test( List net,
              NumericVector label,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun ) {

      int matched = NA_INTEGER;

      List module_ab = as<List>( net["mapfield"] )["ab"];
      List module_a = as<List>( net["module"] )["a"];
      List module_b = as<List>( net["module"] )["b"];

      int category_b = ART::classify( net, module_b["id"], label, activationFun, matchFun );

      int Jmax_a = module_a["Jmax"];

      if ( category_b != NA_INTEGER && Jmax_a != NA_INTEGER ){
        double m_ab = match( module_ab, Jmax_a, category_b, matchFun );

        if ( m_ab >= as<double>( module_ab["rho"] ) ){
          matched = 1;
        }
        else{
          matched = 0;
        }
      }

      return matched;
    }

    List classify( List net,
                   NumericVector d,
                   std::function< double( List, NumericVector, NumericVector ) > activationFun,
                   std::function< double( List, NumericVector, NumericVector ) > matchFun,
                   bool test = false ){

      List module_a = as<List>( net["module"] )["a"];
      List module_ab = as<List>( net["mapfield"] )["ab"];
      List module_b = as<List>( net["module"] )["b"];

      // best matching node in F2a
      int category_a = NA_INTEGER;

      NumericVector predicted( as<int>( module_ab["b_size"] ), NA_REAL );
      NumericVector F1_b( as<NumericMatrix>( module_b["w"] ).cols(), NA_REAL );

      int matched = NA_INTEGER;

      NumericVector a = ART::activation( module_a, d, activationFun );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho_a = module_a["rho"];
      while( !resonance ){
        int Jmax_a = T_j( j );
        double m = ART::match( module_a, d, Jmax_a, matchFun );

        if ( m >= rho_a ){
          // prediction
          category_a = Jmax_a;
          module_a["Jmax"] = Jmax_a;
          predicted = as<NumericMatrix>( module_ab["w"] )( Jmax_a, _ );
          F1_b = recall( module_b, predicted );

          resonance = true;
        }
        else {
          // raise the vigilance of a and continue with the search
          rho_a = std::min( rho_a + as<double>( module_a["epsilon"] ), 1.0 );

          // if it runs out of categories, then it can't find a match
          if ( j == as<int>( module_a["numCategories"] ) - 1 ){
            module_a["Jmax"] = NA_INTEGER;
            resonance = true;
          }
          else{
            // search for the next best matching node in F2a
            j++;
          } // if
        } // else

      } // while resonance

      return List::create( _["category_a"] = category_a,
                           _["F1_b"] = F1_b,
                           _["matched"] = matched );

    }
  }


  void train( List net,
              NumericMatrix x,
              Nullable<NumericVector> vTarget,
              Nullable< NumericMatrix > mTarget,
              std::function< NumericVector( NumericVector ) > codeFun,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > mapfieldUpdateFun ){

    int ep = as<int>( net["maxEpochs"] );
    int nrow = x.rows();
    if ( !isARTMAP( net ) ){
      stop( "The network is not an ARTMAP." );
    }
    if ( !mTarget.isNotNull() && !isSimplified( net ) ){
       stop( "The labels are missing. End running." );
    }
    else if ( !vTarget.isNotNull() && isSimplified( net ) ){
      stop("The labels are missing. End running.");
    }
    for (int i = 1; i <= ep; i++){
      int change = 0;
      std::cout << "Epoch no. " << i << std::endl;

      for (int j = 0; j < nrow; j++){
        if ( isSimplified( net ) )
          change += simplified::learn( net, codeFun( x( j, _ ) ), NumericVector ( vTarget )( j ), activationFun, matchFun, weightUpdateFun );
        else
          change += standard::learn( net, codeFun( x( j, _ ) ), codeFun( NumericMatrix ( mTarget )( j, _ ) ), activationFun, matchFun, weightUpdateFun, mapfieldUpdateFun );
      }

      cout << "Number of changes: " << change << endl;
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
                NumericMatrix x,
                Nullable< NumericVector > vTarget,
                Nullable< NumericMatrix > mTarget,
                std::function< NumericVector( NumericVector ) > codeFun,
                std::function< NumericVector( NumericVector ) > uncodeFun,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun,
                bool test = false ){

    List classified;
    int nrow = x.rows();
    int ncol = x.cols();
    IntegerVector category_a( nrow ), matched( nrow );

    if ( isSimplified( net ) ){
      NumericVector predicted ( nrow );
      for ( int i = 0; i < nrow; i++ ){

          List result = simplified::classify( net, codeFun( x( i,_ ) ), activationFun, matchFun, test );
          category_a( i ) = result( "category_a" );
          predicted( i ) = result( "predicted" );
          if ( test ){
            matched( i ) = simplified::test( predicted( i ), NumericVector( vTarget )( i ) );
          }

      }
      classified = List::create( _["predicted"] = predicted,
                                 _["category_a"] = category_a,
                                 _["matched"] = matched);
    }
    else{
      NumericMatrix predicted( nrow, ncol );

      for ( int i = 0; i < nrow; i++ ){
        List result = standard::classify( net, codeFun( x( i,_ ) ), activationFun, matchFun, test );
        predicted( i, _ ) = uncodeFun( as<NumericVector>( result( "F1_b" ) ) );
        category_a( i ) = result( "category_a" );
        if ( test ){
          matched( i ) = standard::test( net, codeFun( NumericMatrix ( mTarget )( i, _ ) ), activationFun, matchFun );
        }
      }

      classified = List::create( _["predicted"] = predicted,
                                 _["category_a"] = category_a,
                                 _["matched"] = matched );
    }

    return classified;

  }
}


// [[Rcpp::export(.ARTMAP)]]
List newARTMAP ( int numFeatures, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20, bool simplified = false ){
  List net;
  List mapfield;
  if ( !simplified ){
    net = newART( numFeatures, 2, vigilance, learningRate, categorySize, maxEpochs );
    as<List>( net["module"] ).attr( "names" ) = CharacterVector::create( "a", "b" );
  }
  else{
    net = newART( numFeatures, 1, vigilance, learningRate, categorySize, maxEpochs );
    as<List>( net["module"] ).attr( "names" ) = CharacterVector::create( "a" );
  }
  mapfield.push_back( ARTMAP::mapfield( 0, numFeatures, vigilance, learningRate, simplified ), "ab" );
  net.push_back( mapfield, "mapfield" );
  net.attr( "class" ) = "ARTMAP";
  net.attr( "simplified" ) = simplified;
  return net;
}

// [[Rcpp::export(.trainARTMAP)]]
void trainARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue ){

  if ( isFuzzy( net ) ){
    Fuzzy::trainARTMAP( net, x, vTarget, mTarget );
  }
  if ( isHypersphere( net ) ){
    Hypersphere::trainARTMAP( net, x, vTarget, mTarget );
  }

}

// [[Rcpp::export(.predictARTMAP)]]
List predictARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false ){

  List results;
  if ( isFuzzy( net ) ){
    results = Fuzzy::predictARTMAP( net, x, vTarget, mTarget, test );
  }
  if ( isHypersphere( net ) ){
    results = Hypersphere::predictARTMAP( net, x, vTarget, mTarget, test );
  }
  return results;
}

/****************************************************************************
 *
 *  TopoART.cpp
 *  TopoART
 *
 *
 *
 *  Reference:
 *  1. Tscherepanow, M. (2010) "TopoART: A topology learning hierarchical ART network",
 *  Proceedings of the International Conference on Artificial Neural Networks
 *  (ICANN). LNCS, 6354, pp. 157–167.
 *
 ****************************************************************************/

#include <Rcpp.h>
#include "ART.h"
#include "utils.h"
#include "fuzzy.h"
#include "hypersphere.h"
using namespace Rcpp;
using namespace std;


bool isTopoART ( List net ){
  return as<string>( net.attr( "class" ) ).compare( "TopoART" ) == 0;
}

namespace Topo {

  List module( int id, int numFeatures, double vigilance, int phi, double learningRate1, double learningRate2, int categorySize = 200 ){
    IntegerVector n;
    IntegerVector edge;
    List linkedClusters;

    List module = ART::module( id, numFeatures, vigilance, learningRate1, categorySize );  // init with base module
    module.push_back( edge, "edge" );                       // a nx2 matrix that keeps track of the F2 neurons that are linked together
    module.push_back( learningRate1, "beta1" );             // learning rate for the highest activated neuron
    module.push_back( learningRate2, "beta2" );             // learning rate for the second neuron with the second highest activation after the best matching neuron
    module.push_back( n, "n" );                             // counter
    module.push_back( phi, "phi" );                         // counter threshold
    module.push_back( linkedClusters, "linkedClusters" );
    return module;
  }


  void newCategory( List net, NumericVector x ){
    NumericVector w = clone( x );
    int numCategories = net["numCategories"];

    // update node
    if ( numCategories == 0 ){
      IntegerVector n = IntegerVector::create( 1 );
      net["n"] = n;
      as<NumericMatrix>( net["w"] )( 0,_ ) = w;
    }
    else{
      IntegerVector n = as<IntegerVector>( net["n"] );
      n.push_back( 1 );
      net["n"] = n;
      NumericMatrix wm = as<NumericMatrix>( net["w"] );
      int nrow = wm.nrow();
      int ncol = wm.ncol();
      if ( numCategories < wm.rows() ){
        wm( numCategories,_ ) = w;
      }
      else{
        // need to add more rows to the weight matrix
        NumericMatrix newWeight ( nrow + as<int>( net["capacity"] ), ncol );
        for (int i = 0; i < nrow; i++){
          for (int j = 0; j < ncol; j++){
            newWeight ( i,j ) = wm( i,j );
          }
        }
        newWeight ( nrow,_ ) = w;
        net["w"] = newWeight;
      }

    }
    net["numCategories"] = numCategories + 1;
  }

  double rho ( List net, int moduleId ){
    List module = as<List>( net["module"] )[ moduleId ];
    if ( moduleId - 1 >= 0 ){
      return 0.5*( as<double>( module["rho"] ) + 1 );
    }
    return as<double>( module["rho"] );
  }

  IntegerVector link ( IntegerVector edge, int bm, int sbm ) {

    int s = edge.size();
    if ( s == 0 ){
      edge.push_back( bm );
      edge.push_back( sbm );
    }
    else{
      // check if neuronIds of bm and smb are already in the edge vector
      bool linked = false;
      for ( int j = 0; j < s; j += 2 ){
        if ( ( edge( j ) == bm && edge( j+1 ) == sbm ) || ( edge( j ) == sbm && edge( j+1 ) == bm ) ){
          linked = true;
          break;
        }
      }
      if ( !linked ){
        edge.push_back( bm );
        edge.push_back( sbm );
      }
    }
    return edge;
  }

  IntegerVector counterUpdate ( IntegerVector counter, int nodeIndex ){
    counter( nodeIndex ) = counter( nodeIndex ) + 1;
    return counter;
  }

  int removeF2Nodes ( List module ){
    int change = 0;
    vector <int> indices;
    IntegerVector newIndices;
    int idx = 0;
    IntegerVector oldn = as<IntegerVector>( module["n"] );
    int l = oldn.size();
    int phi = as<int>( module["phi"] );

    if ( l == 0 ){
      // no F2 categories
      return 0;
    }

    // get all neuron indices that have counts >= phi
    for ( int k = 0; k < l; k++ ){
      if ( oldn( k ) >= phi ){
        indices.push_back( k );
        newIndices.push_back( idx );
        idx++;
      }
      else{
        newIndices.push_back( -1 ); // set this removed index to -1
        change++;
      }
    }
    int s = indices.size();
    if ( s > 0 ){
      // save the permanent nodes
      IntegerVector n( s ); // new counter vector
      NumericMatrix oldw = as<NumericMatrix>( module["w"] );
      int weightSize = ( s/as<int>( module["capacity"] ) + 1 ) * as<int>( module["capacity"] );
      NumericMatrix w( weightSize, oldw.cols() ); // new weight matrix
      // loop through all permanent neurons
      for ( int i = 0; i < s; i++ ){
        n( i ) = oldn( indices[i] );
        w( i,_ ) = oldw( indices[i], _ );
      }

      // remove edges: The edge neurons are simply the order index of the count (n) vector
      // create a matrix that holds the indices of the old categories and the indices of the survived categories
      // get all neuron indices that have counts >= phi

      IntegerVector olde = as<IntegerVector>( module["edge"] );
      int z = olde.size();
      IntegerVector e; // new edge vector
      if ( z > 0 ){
        // for each neuron that is now permanent, keep them in w and n
        for ( int j = 0; j < z; j += 2 ){ // for each edge (neuron pair)
          int neuron1 = -1;
          int neuron2 = -1;
          // for each neuron index that is now permanent
          // if neuron j is permanent, keep it but change its position to k due to the removal of
          // the node candidate. To get the new position, find from the new counter vector n
          // which index position this neuron in the old counter position is now located.
          if ( find( indices.begin(), indices.end(), olde( j ) ) != indices.end() ){
            neuron1 = newIndices[olde( j )];
          }
          if ( find( indices.begin(), indices.end(), olde( j+1 ) ) != indices.end() ){
            neuron2 = newIndices[olde( j+1 )];
          }
          if ( neuron1  != -1 && neuron2 != -1 ){
            // both neurons are permanent, so keep their edge
            e.push_back( neuron1 );
            e.push_back( neuron2 );
          }

        }
        module["edge"] = e;
      }
      else{
        IntegerVector edge;
        module["edge"] = edge;
      }

      module["n"] = n;
      module["w"] = w;
      module["numCategories"] = indices.size();
    }
    else{
      // there is no permanent node to save
      IntegerVector n;
      IntegerVector edge;
      module["n"] = n;
      int cols = as<NumericMatrix>( module["w"] ).cols();
      NumericMatrix w = no_init( as<int>( module["capacity"] ), cols );
      module["w"] = w;
      module["numCategories"] = 0;
      module["edge"] = edge;
    }

    return change;
  }

  int getLinkedCluster( List module, int category ){
    List linkedClusters = as<List>( module["linkedClusters"] );
    int cluster = -1;
    int l = linkedClusters.length();
    for ( int i = 0; i < l; i++ ){
      NumericVector c = as<NumericVector>( linkedClusters[i] );
      if ( find( c.begin(), c.end(), category ) != c.end() ){
        cluster = i;
        break;
      }
    }
    return cluster;
  }

  void learn ( List net,
               int id,
               NumericVector d,
               NumericVector change,
               std::function< double( List, NumericVector, NumericVector ) > activationFun,
               std::function< double( List, NumericVector, NumericVector ) > matchFun,
               std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun ){

    List module = as<List>( net["module"] )[id];

    int nc = module["numCategories"];
    if ( nc == 0 ){
      newCategory( module, d );
      change[id]++;
    }
    else{

      NumericVector a = ART::activation( module, d, activationFun );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho_a = rho( net, id );
      NumericVector matchIndex; // temporarily holds the indices of the bm and sbm neurons

      while( !resonance ){

        int J_max = T_j( j );
        double m = ART::match( module, d, J_max, matchFun );

        if ( m >= rho_a ){
          matchIndex.push_back( J_max );
          if ( matchIndex.size() == 1 ){
            module["beta"] = module["beta1"];
          }
          else if ( matchIndex.size() == 2 )
            module["beta"] = module["beta2"];

          change[id] += ART::weightUpdate( module, d, J_max, weightUpdateFun );
          module["n"] = counterUpdate( module["n"], J_max );

          // both bm and sbm neurons are found, then link them together
          if ( matchIndex.size() == 2 ){
            module["edge"] = link( module["edge"], matchIndex( 0 ), matchIndex( 1 ) );
            resonance = true;
          }
          else{
            // matchIndex.size() == 1, so move up to the next module if count >= phi.
            // Once return to this module, continue to search for the sbm
            int count = as<IntegerVector>( module["n"] )( j );
            if ( count >= as<int>( module["phi"] ) ) {
              if ( ( id+1 ) < as<int>( net["numModules"] ) ){
                // match >= rho_a and count > phi, then activate net b
                learn( net, id+1, d, change, activationFun, matchFun, weightUpdateFun );
              }
            }
            if ( j < nc - 1 ){
              j++; // search for the sbm neuron
            }
            else{
              resonance = true;
            }
          }

        } // match >= rho_a
        else{
          if ( j == nc - 1 ){ // all F2 neurons have been enumerated

            if ( matchIndex.size() == 0 ){
              // We haven't found a bm neuron, so create a new neuron
              newCategory( module, d );
              change[id]++;
            }
            // True when 1) no bm neuron is found, or 2) the bm has been found but not the sbm
            resonance = true;
          }
          j++;
        } // match
      } // while resonance
    } // if
  }


  void train( List net,
              NumericMatrix x,
              std::function< NumericVector( NumericVector ) > codeFun,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun ){
    cout << "Training TopoART" << endl;

    int tau = 0;
    int epoch;
    int maxEpochs = as<int>( net["maxEpochs"] );
    int numModules = as<int>( net["numModules"] );
    int nrow = x.rows();

    NumericVector change( numModules );
    bool flag = false; // controls when to terminate learning
    bool complete = false; // if learning completes before the maximum epoch is reached

    for ( epoch = 1; epoch <= maxEpochs; epoch++ ){
      cout << "Epoch no. " << epoch << endl;
      if ( epoch > 1 ) flag = true;
      if ( isTopoART( net ) ){
        for ( int i = 0; i < nrow; i++ ){
          learn( net, 0, codeFun( x( i, _ ) ), change, activationFun, matchFun, weightUpdateFun );
          tau++;
          if ( tau == as<int>( net["tau"] ) ){
            // Reach the end of the learning cycle. Remove node candidates.
            for ( int j = 0; j < numModules; j++ ){
              List module = as<List>( net["module"] )[j];
              change[j] -= removeF2Nodes( module );
            }
            if ( flag ){
              for ( int j = 0; j < numModules; j++ ){
                cout << "ID " << j << " Number of changes: " << change( j ) << endl;
              }
              if ( sum( change )  == 0 ) {
                net["epochs"] = epoch;
                complete = true;
                break;
              } else{
                for ( int j = 0; j < numModules; j++ ){
                  change[j] = 0;
                }
              }
            }
            tau = 0;
            flag = false;
          }
        }
      }

      if ( complete ) {
        break;
      }
    }

    // subset weight matrix
    int l = as<List>( net["module"] ).length();
    for ( int i = 0; i < l; i++ ){
      List module = as<List>( net["module"] )[i];
      removeF2Nodes( module );  // remove all node candidates one last time
      module["w"] = subsetRows( module["w"], module["numCategories"] );
      module["edge"] = vectorToMatrix( module["edge"], 2, as<NumericVector>( module["edge"] ).size()/2 );
      module["linkedClusters"] = ::linkClusters( module["edge"], seq( 0,as<NumericMatrix>( module["w"] ).rows() - 1 ) );
    }

  }

  NumericVector classify( List net,
                          int id,
                          NumericVector d,
                          std::function< double( List, NumericVector, NumericVector ) > activationFun,
                          std::function< double( List, NumericVector, NumericVector ) > matchFun,
                          NumericVector classified ){

    List module = as<List>( net["module"] )[id];
    int category = NA_INTEGER;

    NumericVector a = ART::activation( module, d, activationFun );
    NumericVector T_j = sortIndex( a );
    bool resonance = false;
    int j = 0;

    while(!resonance){
      int J_max = T_j( j );
      double m = ART::match( module, d, J_max, matchFun );
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

  List predict( List net,
                int id,
                NumericMatrix x,
                std::function< NumericVector( NumericVector ) > codeFun,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun ){
    int nrow = x.rows();
    NumericVector category( nrow );
    NumericVector linkedCluster( nrow );

    for (int i = 0; i < nrow; i++){

        int result = ART::classify( net, id, codeFun( x( i,_ ) ), activationFun, matchFun );
        int cluster = getLinkedCluster( as<List>( net["module"] )[id], result );
        category( i ) = result;
        if ( cluster == -1 ){
          linkedCluster( i ) = NA_INTEGER;
        }
        else{
          linkedCluster( i ) = cluster;
        }

    }
    List classified = List::create( _["category"] = category,
                                    _["linkedCluster"] = linkedCluster);

    return classified;
  }

}

// [[Rcpp::export(.TopoART)]]
List TopoART ( int numFeatures, int num = 2, double vigilance = 0.9, double learningRate1 = 1.0, double learningRate2 = 0.6, int tau = 100, int phi = 6, int categorySize = 200, int maxEpochs = 20 ){
  List net = List::create( _["numModules"] = num,       // number of topoART modules
                           _["epochs"] = 0,             // total number of epochs required to learn
                           _["maxEpochs"] = maxEpochs,  // maximum number of epochs
                           _["tau"] = tau );            // number of learning cycles required before the F2 nodes with low counts are removed

  List modules;
  for (int i = 0; i < num; i++){
    modules.push_back( Topo::module( i, numFeatures, vigilance, phi, learningRate1, learningRate2, categorySize ) );
  }

  net.push_back( modules, "module" );

  net.attr( "class" ) = CharacterVector::create( "TopoART" );
  return net;
}

// [[Rcpp::export(.topoTrain)]]
void topoTrain( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue ){

  if ( isFuzzy( net ) ){
    Fuzzy::trainART( net, x );
  }
  if ( isHypersphere( net ) ){
    Hypersphere::trainART( net, x );
  }

}


// [[Rcpp::export(.topoPredict)]]
List topoPredict( List net, int id, NumericMatrix x ){
  List results;
  if ( isFuzzy( net ) ){
    results = Fuzzy::predictART( net, id, x );
  }
  if ( isHypersphere( net ) ){
    results = Hypersphere::predictART( net, id, x );
  }
  return results;
};

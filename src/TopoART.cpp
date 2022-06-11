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

  List module( int id, double vigilance, int phi, double learningRate1, double learningRate2, int categorySize = 200 ){
    IntegerVector n;
    IntegerVector edge;
    List linkedClusters;
    
    List module = ART::module(id, vigilance, learningRate1, categorySize );  // init with base module
    module.push_back( edge, "edge" );                       // a nx2 matrix that keeps track of the F2 neurons that are linked together
    module.push_back( learningRate1, "beta1" );             // learning rate for the highest activated neuron
    module.push_back( learningRate2, "beta2" );             // learning rate for the second neuron with the second highest activation after the best matching neuron
    module.push_back( linkedClusters, "linkedClusters" );   // the linked clusters
    module.push_back( n, "n" );                             // accumulator for noise filtering
    module.push_back( phi, "phi" );                         // counter threshold
    ART::addJmax( module, -1 );                             // sbm Jmax
    return module;
  }
  
  IntegerVector getEdgeVector( List module ){
    return module["edge"];
  }
  
  void setEdgeVector( List module, IntegerVector edge ){
    module["edge"] = edge;
  }
  
  int getTau( List net ){
    return net["tau"];
  }
  
  
  int getCounterThreshold( List module ){
    return module["phi"];
  }
  
  double getLearningRate( List module, int learningRateIndex ){
    if ( learningRateIndex == 1 ){
      return module["beta1"];
    }
    else if ( learningRateIndex == 2 ){
      return module["beta2"];
    }
    return module["beta"];
  }
  
  IntegerVector link ( IntegerVector edge, int bm, int sbm ) {
    
    int s = edge.size();
    if ( s == 0 ){
      edge.push_back( bm );
      edge.push_back( sbm );
    }
    else{
      // check if neuronIds of bm and sbm are already in the edge vector
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
  
  int getAccumulator( List module, int weightIndex ){
    return as<IntegerVector>( module["n"] )( weightIndex );
  }
  
  IntegerVector getAccumulatorVector( List module ){
    return module["n"];
  }
  
  void accumulatorUpdate ( List module, int nodeIndex ){
    IntegerVector n = as<IntegerVector>( module["n"] );
    n( nodeIndex ) = n( nodeIndex ) + 1;
    
    module["n"] = n;
  }
  
  void setAccumulatorVector( List module, IntegerVector v ){
    module["n"] = v;
  }
  
  void removeF2Nodes ( List module ){
    
    vector <int> indices;
    IntegerVector newIndices;
    int idx = 0;
    IntegerVector oldn = ART::getCounterVector( module );
    IntegerVector oldchange = ART::getChangeVector( module );
    IntegerVector olda = getAccumulatorVector( module );
    int l = oldn.size();
    int phi = getCounterThreshold( module );
    
    // if the module is not empty without nodes
    if ( l > 0 ){
      // get all neuron indices that have counts >= phi
      for ( int k = 0; k < l; k++ ){
        if ( olda( k ) >= phi ){
          indices.push_back( k );
          newIndices.push_back( idx );
          idx++;
        }
        else{
          // the count < phi, so set it to -1 and increment change due to this noisy node
          newIndices.push_back( -1 ); // set this removed index to -1
          
        }
      }
      int s = indices.size();
      if ( s > 0 ){
        // save the permanent nodes
        IntegerVector n( l ); // new counter vector
        IntegerVector a( l ); // new accumulator vector
        IntegerVector c( l ); // new change vector
        NumericMatrix oldw = ART::getWeightMatrix( module );
        int numNodes = ART::getCapacity( module );
        int weightSize = ( s/numNodes + 1 ) * numNodes;
        NumericMatrix w( weightSize, oldw.cols() ); // new weight matrix
        // loop through all permanent neurons
        for ( int i = 0; i < s; i++ ){
          n( i ) = oldn( indices[i] );
          a( i ) = olda( indices[i] );
          w( i,_ ) = oldw( indices[i], _ );
          c( i ) = oldchange( indices[i] );
        }
        
        // remove edges: The edge neurons are simply the order index of the count (n) vector
        // create a matrix that holds the indices of the old categories and the indices of the survived categories
        // get all neuron indices that have counts >= phi
        
        IntegerVector olde = getEdgeVector( module );
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
          
          setEdgeVector( module, e );
        }
        else{
          IntegerVector edge;
          setEdgeVector( module, edge );
        }
        
        ART::setCounterVector( module, n );
        setAccumulatorVector( module, a );
        ART::setChangeVector( module, c );
        ART::setWeightMatrix( module, w );
        ART::setNumCategories( module, indices.size() );
      }
      else{
        // there is no permanent node to save
        IntegerVector n( ART::getCapacity( module ) );
        IntegerVector c( ART::getCapacity( module ) );
        IntegerVector a( ART::getCapacity( module ) );
        ART::setCounterVector( module, n );
        setAccumulatorVector( module, a );
        ART::setChangeVector( module, c );
        IntegerVector edge;
        int cols = ART::getWeightMatrix( module ).cols();
        NumericMatrix w = no_init( ART::getCapacity( module ), cols );
        ART::setWeightMatrix( module , w );
        ART::setNumCategories( module, 0 );
        setEdgeVector( module, edge );
      }
    }
  }
  
  int getLinkedClusters( List module, int category ){
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
  
  void setLinkedClusters( List module, List linkedClusters ){
    module["linkedClusters"] = linkedClusters;
  }
  
  // rho: If there is just one module, then return the rho as is. If there are more than
  // one module in the hierarchy, then the next module in the hierarchy will have 
  // rho = 0.5*(rho + 1) with rho being the vigilance value of the previous module.
  double rho ( double rho, int moduleId ){
    
    for ( int i = 1; i <= moduleId; i++ ){
      rho = 0.5*( rho + 1 );
    }
    
    return rho;
  }
  
  void setJmax( List module, int J, int matchIndex = 0 ){
    IntegerVector v = module["Jmax"];
    v[matchIndex] = J;
    module["Jmax"] = v;
  }
  
  int getJmax( List module, int matchIndex = 0 ){
    return as<IntegerVector>( module["Jmax"] )[matchIndex];
  }
  
  void init( IModel &model ){
    ART::init( model );
    int n = ART::getNumModules( model.net );
    for ( int i = 0; i < n; i++ ){
      List module = ART::getModule( model.net, i );
      IntegerVector a( ART::getCapacity( module ) );
      setAccumulatorVector( module, a );
    }
  }
  
  void newCategory( IModel &model, List module, NumericVector x ){
    int numCategories = ART::getNumCategories( module );
    int newCategoryIndex = numCategories;
    IntegerVector v = getAccumulatorVector( module );
    if ( numCategories == v.length() ){
      IntegerVector n = appendVector( v, ART::getCapacity( module ) );
      setAccumulatorVector( module, n );
    }
    accumulatorUpdate( module, newCategoryIndex );
    ART::newCategory( model, module, x );
    
  }
  
  void weightUpdate( IModel &model, List module, int weightIndex, NumericVector x, int bmIndex ){
    ART::setLearningRate( module, getLearningRate( module, bmIndex ) );
    ART::weightUpdate( model, module, weightIndex, x );
  }
  
  void learn ( IModel &model, 
               int id,
               NumericVector d ){
    
    List module = ART::getModule( model.net, id );
    
    int nc = ART::getNumCategories( module );
    if ( nc == 0 ){
      newCategory( model, module, d );
    }
    else{
      
      NumericVector a = ART::activation( model, module, d );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho_a = ART::getRho( module );
      int matchCount = 0; // temporarily holds the indices of the bm and sbm neurons
      
      while( !resonance ){
        
        int J_max = T_j( j );
        double m = ART::match( model, module, J_max, d );
        
        if ( m >= rho_a ){
          setJmax( module, J_max, matchCount );
          matchCount++;
          weightUpdate( model, module, J_max, d, matchCount );
          
          if ( matchCount == 1 ){
            ART::counterUpdate( module, J_max );
            accumulatorUpdate( module, J_max );
          }
          
          // both bm and sbm neurons are found, then link them together
          if ( matchCount == 2 ){
            setEdgeVector( module, link( getEdgeVector( module ), getJmax( module, 0 ), getJmax( module, 1 ) ) );
            resonance = true;
          }
          else{
            // matchIndex.size() == 1, so move up to the next module if count >= phi.
            // Once return to this module, continue to search for the sbm
            int count = getAccumulator( module, J_max );
            if ( count >= getCounterThreshold( module ) ) {
              if ( ART::hasMoreModules( model.net, id ) ){
                // match >= rho_a and count > phi, then activate net b
                learn( model, id+1, d );
              }
            }
            if ( j < nc - 1 ){
              j++; 
            }
            else{
              resonance = true;
            }
          }
        } // match >= rho_a
        else{
          if ( j == nc - 1 ){ // all F2 neurons have been enumerated
            
            if ( matchCount == 0 ){
              // We haven't found a bm neuron, so create a new neuron
              newCategory( model, module, d );
            }
            // True when 1) no bm neuron is found, or 2) the bm has been found but not the sbm
            resonance = true;
          }
          j++;
        } // match
      } // while resonance
    } // if
  }
  
  
  void train( IModel &model,
              NumericMatrix x ){
    
    init( model );
    
    cout << "Training TopoART" << endl;
    
    int tau = 0;
    int epoch;
    int maxEpochs = ART::getMaxEpochs( model.net );
    int numModules = ART::getNumModules( model.net );
    int nrow = x.rows();
    
    bool flag = false; // controls when to terminate learning
    bool complete = false; // if learning completes before the maximum epoch is reached
    
    for ( epoch = 1; epoch <= maxEpochs; epoch++ ){
      cout << "Epoch no. " << epoch << endl;
      
      if ( isTopoART( model.net ) ){
        for ( int i = 0; i < nrow; i++ ){
          learn( model, 0, model.processCode( x( i, _ ) ) );
          tau++;
          if ( tau == getTau( model.net ) ){
            // Reach the end of the learning cycle. Remove node candidates.
            for ( int j = 0; j < numModules; j++ ){
              List module = ART::getModule( model.net, j );
              removeF2Nodes( module );
            }
            
            tau = 0;
            flag = false;
          }
        }
        for ( int j = 0; j < numModules; j++ ){
          cout << "ID " << j << " Number of changes: " << ART::getModuleChange( ART::getModule( model.net, j ) ) << endl;
        }
        if ( ART::getTotalChange( model.net ) == 0 ) {
          ART::setEpoch( model.net, epoch );
          complete = true;
          break;
        } else{
          for ( int j = 0; j < numModules; j++ ){
            List module = ART::getModule( model.net, j );
            ART::changeReset( module );
            ART::counterReset( module );
          }
        }
      }
      
      if ( complete ) {
        break;
      }
    }
    
    // subset weight matrix
    int l = ART::getNumModules( model.net );
    for ( int i = 0; i < l; i++ ){
      List module = ART::getModule( model.net, i );
      removeF2Nodes( module );  // remove all node candidates one last time
      int numCategories = ART::getNumCategories( module );
      ART::setWeightMatrix( module, subsetRows( ART::getWeightMatrix( module ), numCategories ) );
      ART::setCounterVector( module, subsetVector( ART::getCounterVector( module ), numCategories ) );
      setAccumulatorVector( module, subsetVector( getAccumulatorVector( module ), numCategories ) );
      ART::setChangeVector( module, subsetVector( ART::getChangeVector( module ), numCategories ) );
      IntegerVector edges = getEdgeVector( module );
      if ( edges.size() > 0 ){
        setEdgeVector( module, as<IntegerVector>( vectorToMatrix( as<NumericVector>( edges ), 2, edges.size()/2 ) ) );
        setLinkedClusters( module, linkClusters( getEdgeVector( module ), seq( 0, ART::getWeightMatrix( module ).rows() - 1 ) ) );
      }
    }
    
  }
  
  NumericVector classify( IModel &model,
                          int id,
                          NumericVector d ){
    
    List module = ART::getModule( model.net, id );
    int category = NA_INTEGER;
    
    NumericVector a = ART::activation( model, module, d );
    NumericVector T_j = sortIndex( a );
    bool resonance = false;
    int j = 0;
    
    while(!resonance){
      int J_max = T_j( j );
      double m = ART::match( model, module, J_max, d );
      if ( m >= ART::getRho( module )){
        setJmax( module, J_max, 0 );
        category = J_max;
        resonance = true;
      }
      else{
        if ( j  == ART::getNumCategories( module ) - 1 ){
          setJmax( module, category, 0 );
          resonance = true;
        }
        else{
          j++;
        } // if
      } // match
    } // while resonance
    
    return category;
  }
  
  List predict( IModel &model,
                int id,
                NumericMatrix x ){
    int nrow = x.rows();
    NumericVector category( nrow );
    NumericVector linkedCluster( nrow );
    
    for (int i = 0; i < nrow; i++){
      
      int result = ART::classify( model, id, model.processCode( x( i,_ ) ) );
      int cluster = getLinkedClusters( ART::getModule( model.net, id ), result );
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
List TopoART ( int dimension, int num = 2, double vigilance = 0.9, double learningRate1 = 1.0, double learningRate2 = 0.6, int tau = 100, int phi = 6, int categorySize = 200, int maxEpochs = 20 ){
  if ( vigilance < 0.0 || vigilance > 1.0 ){
    stop( "The vigilance value must be between 0 and 1.0." );
  }
  if ( learningRate1 < 0.0 || learningRate1 > 1.0 ){
    stop( "The learningRate1 value must be between 0 and 1.0." );
  }
  if (learningRate2 < 0.0 || learningRate2 > 1.0){
    stop( "The learningRate2 value must be between 0 and 1.0." );
  }
  if ( num < 2 ){
    stop( "The num value must be at least 2." );
  }
  if ( tau < 0 ){
    stop( "The tau value must be at least 0." );
  }
  if ( phi < 0 ){
    stop( "The phi value must be at least 0." );
  }
  if ( categorySize < 1 ){
    stop( "The categorySize value must be greater than 0." );
  }
  if ( maxEpochs < 1 ){
    stop( "The maxEpochs value must be greater than 0." );
  }
  if ( dimension < 1 ){
    stop( "The dimension value must be greater than 0." );
  }
  
  
  List net = List::create( _["numModules"] = num,       // number of topoART modules
                           _["dimension"] = dimension,  // number of features
                           _["epochs"] = 0,             // total number of epochs required to learn
                           _["maxEpochs"] = maxEpochs,  // maximum number of epochs
                           _["tau"] = tau );            // number of learning cycles required before the F2 nodes with low counts are removed
  
  List modules;
  for (int i = 0; i < num; i++){
    vigilance = Topo::rho( vigilance, i );
    if ( vigilance > 0 ){
      modules.push_back( Topo::module( i, vigilance, phi, learningRate1, learningRate2, categorySize ) );
    }
    else{
      net["numModules"] = as<int>( net["numModules"] ) - 1;
      cout << "Module Id " << i << " cannot be created because its vigilance will be 0 or negative." << endl;
    }
    
  }
  
  net.push_back( modules, "module" );
  
  net.attr( "class" ) = CharacterVector::create( "TopoART" );
  return net;
}

// [[Rcpp::export(.topoTrain)]]
void topoTrain( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue ){
  IModel *model;
  if ( isFuzzy( net ) ){
    model = new Fuzzy( net );
    
  }
  if ( isHypersphere( net ) ){
    model = new Hypersphere( net, x );
  }
  
  Topo::init( *model );
  Topo::train( *model, x );
  
  delete model;
  
}


// [[Rcpp::export(.topoPredict)]]
List topoPredict( List net, int id, NumericMatrix x ){
  IModel *model;
  if ( isFuzzy( net ) ){
    model = new Fuzzy( net );
  }
  if ( isHypersphere( net ) ){
    model = new Hypersphere( net );
  }
  List results = Topo::predict( *model, id, x );
  delete model;
  
  return results;
};

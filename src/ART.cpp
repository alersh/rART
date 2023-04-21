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


namespace ART {

  List module ( int id, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100 ){
    NumericMatrix w;
    IntegerVector n;
    IntegerVector change;
    IntegerVector jmax;  
    jmax.push_back( -1 );  // one element of -1
    List module = List::create( _["id"] = id,                   // module id
                                _["weightDimension"] = 0,       // the number of dimensions in the weight
                                _["capacity"] = categorySize,   // number of category to create when the module runs out of categories to match
                                _["numCategories"] = 0,         // number of categories created during learning
                                _["alpha"] = 0.001,             // activation function parameter
                                _["epsilon"] = 0.000001,        // match function parameter
                                _["w"] = w,                     // top-down weights
                                _["rho"] = vigilance,           // vigilance parameter
                                _["beta"] = learningRate,       // learning parameter
                                _["Jmax"] = jmax,               // the node index with the highest activation and the best match
                                _["counter"] = n,               // counter
                                _["change"] = change            // keep track of the number of changes in each node
    );
    
    module.attr( "class" ) = "ART";
    return module;
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
  
  List create ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20 ){
    if ( vigilance < 0.0 || vigilance > 1.0 ){
      stop( "The vigilance value must be between 0 and 1.0." );
    }
    if ( learningRate < 0.0 || learningRate > 1.0 ){
      stop( "The learningRate value must be between 0 and 1.0." );
    }
    if ( num < 1 ){
      stop( "The num value must be greater than 0." );
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
    
    
    List net = List::create( _["numModules"] = num,       // number of ART modules
                             _["dimension"] = dimension,  // number of features
                             _["epochs"] = 0,             // total number of epochs required to learn
                             _["maxEpochs"] = maxEpochs,  // maximum number of epochs
                             _["activeModule"] = -1,      // an index that keeps track of which module is currently active
                             _["init"] = 0                // whether the net is initialized. 0 = not initialized; 1 = initialized
    );
    
    List modules;
    for ( int i = 0; i < num; i++ ){
      vigilance = ART::rho( vigilance, i );
      if ( vigilance > 0 ){
        modules.push_back( ART::module( i, vigilance, learningRate, categorySize ) );
      }
      else{
        net["numModules"] = as<int>( net["numModules"] ) - 1;
        std::cout << "Module Id " << i << " cannot be created because its vigilance will be 0 or negative." << std::endl;
      }
    }
    
    net.push_back( modules, "module" );
    
    net.attr( "class" ) = CharacterVector::create( "ART" );
    return net;
  }
  
  int getID( List module ){
    return module["id"];
  }

  int getDimension( List net ){
    return net["dimension"];
  }
  
  int getJmax( List module, int matchIndex = 0 ){
    return as<IntegerVector>( module["Jmax"] )[matchIndex];
  }
  
  void setJmax( List module, int J, int matchIndex = 0 ){
    IntegerVector v = module["Jmax"];
    v[matchIndex] = J;
    module["Jmax"] = v;
  }
  
  void addJmax( List module, int J ){
    IntegerVector v = module["Jmax"];
    v.push_back( J );
    module["Jmax"] = v;
  }
  
  int getMaxEpochs( List net ){
    return net["maxEpochs"];
  }
  
  void setEpoch( List net, int epoch ){
    net["epochs"] = epoch;
  }
  
  int getNumModules( List net ){
    return net["numModules"];
  }
  
  List getModule( List net, int moduleID ) {
    return as<List>( net["module"] )[moduleID];
  }
  
  void setModule( List net, List module ){
    as<List>( net["module"] )[getID( module )] = module;
  }
  
  bool hasMoreModules( List net, int currentModuleID ){
    return ( currentModuleID+1 ) < ART::getNumModules( net ); 
  }
  
  List nextModule( List net ){
    int numModules = getNumModules( net );
    if ( hasMoreModules ( net, net["activeModule"] ) ){
      return getModule( net, as<int>( net["activeModule"] ) + 1 );
    }
    return NULL;
  }
  
  void incChange( List module, int index ){
    as<IntegerVector>( module["change"] )[index]++;
  }
  
  int getModuleChange( List module ){
    return sum( as<IntegerVector>( module["change"] ) );
  }

  int getTotalChange( List net ){
    int n = getNumModules( net );
    int s = 0;
    for ( int i = 0; i < n; i++ ){
      List module = getModule( net, i );
      s += getModuleChange( module );
    }
    return s;
  }

  IntegerVector getChangeVector( List module ){
    return module["change"];
  }
  
  void setChangeVector( List module, IntegerVector v ){
    module["change"] = v;
  }
  
  void changeReset( List module ){
    IntegerVector c = as<IntegerVector>( module["change"] );
    int l = c.length();
    for ( int i = 0; i < l; i++ ){
      c( i ) = 0;
    }
    
    module["change"] = c;
  }

  int getCapacity( List module ){
    return module["capacity"];
  }
  
  NumericMatrix getWeightMatrix( List module ){
    return module["w"];
  }
  void setWeightMatrix( List module, NumericMatrix w ){
    module["w"] = w;
  }

  int getWeightDimension( List module ){
    return module["weightDimension"];
  }

  void setWeightDimension( List module, int dimension ){
    module["weightDimension"] = dimension;
  }
  
  NumericVector getWeight( List module, int weightIndex ) {
    return getWeightMatrix( module )( weightIndex, _ );
  }
  void setWeight( List module, int weightIndex, NumericVector w ) {
    NumericMatrix wm = as<NumericMatrix>( module["w"] );
    wm( weightIndex, _ ) = w;
  }
  
  double getRho( List module ) {
    return module["rho"];
  }
  void setRho( List module, double rho ) {
    module["rho"] = rho;
  }
  
  int getNumCategories( List module ){
    return module["numCategories"];
  }
  
  void setNumCategories( List module, int x ){
    module["numCategories"] = x;
  }
  
  double getLearningRate( List module ){
    return module["beta"];
  }
  
  void setLearningRate( List module, double learningRate ){
    module["beta"] = learningRate;
  }
  
  int getCounter( List module, int weightIndex ){
    return as<IntegerVector>( module["counter"] )( weightIndex );
  }
  
  IntegerVector getCounterVector( List module ){
    return module["counter"];
  }
  
  void counterUpdate ( List module, int nodeIndex ){
    IntegerVector n = as<IntegerVector>( module["counter"] );
    n( nodeIndex ) = n( nodeIndex ) + 1;
    
    module["counter"] = n;
  }
  
  void counterReset ( List module ){
    IntegerVector n = as<IntegerVector>( module["counter"] );
    int l = n.length();
    for ( int i = 0; i < l; i++ ){
      n( i ) = 0;
    }
    
    module["counter"] = n;
  }
  
  void setCounterVector( List module, IntegerVector v ){
    module["counter"] = v;
  }
  
  double getAlpha( List module ){
    return module["alpha"];
  }
  
  double getEpsilon( List module ){
    return module["epsilon"];
  }
  
  void initModule( List module, int weightDimension ){
    int size = ART::getCapacity( module );
    setWeightDimension( module, weightDimension );
    
    NumericMatrix w = no_init( size, weightDimension );
    setWeightMatrix( module, w );
    
    IntegerVector n( size );
    setCounterVector( module, n);
    
    IntegerVector c(size);
    setChangeVector( module, c );
  }
  
  void init( IModel &model ){
    int n = ART::getNumModules( model.net );
    for ( int i = 0; i < n; i++ ){
      List module = ART::getModule( model.net, i );
      initModule( module, model.getWeightDimension( getDimension( model.net ) ) );
    }
    model.net["init"] = 1;
  }

  bool isInitialized( List net ){
    return net["init"];
  }
  
  NumericVector activation( IModel &model, List module, NumericVector x ){
    
    int nc = getNumCategories( module );
    NumericVector a( nc );
    
    for ( int k = 0; k < nc; k++ ){
      NumericVector w = getWeight( module, k );
      a[k] = model.activation( module, x, w );
    }
    
    return a;
  }
  
  double match( IModel &model, List module, int weightIndex, NumericVector x ){
    NumericVector w = getWeight( module, weightIndex );
    double a = model.match( module, x, w );
    return a;
  }
  
  
  void weightUpdate( IModel &model, List module, int weightIndex, NumericVector x ){
    
    NumericVector w_old = getWeight( module, weightIndex );
    NumericVector w_new = model.weightUpdate( module, getLearningRate( module ), x, w_old );
    setWeight( module, weightIndex, w_new );
    double s = sum( abs ( w_old - w_new ) );
    if ( s > 0.0000001 ){
      incChange( module, weightIndex );
    }
    
  }
  
  void newCategory( IModel &model, List module, NumericVector x ){
    
    int numCategories = getNumCategories( module );
    int newCategoryIndex = numCategories;
    NumericMatrix wm = getWeightMatrix( module );
    if ( numCategories == wm.rows() ){
      // reached the max capacity, so add more rows
      NumericMatrix newWeight = appendRows( wm, getCapacity( module ) );
      setWeightMatrix( module, newWeight );
      
      IntegerVector n = appendVector( getCounterVector( module ), getCapacity( module ) );
      setCounterVector( module, n );
      
      IntegerVector c = appendVector( getChangeVector( module ), getCapacity( module ) );
      setChangeVector( module, c );
    }
    setWeight( module, newCategoryIndex, model.newWeight( x ) );
    counterUpdate( module, newCategoryIndex );
    incChange( module, newCategoryIndex );
    setNumCategories( module, newCategoryIndex + 1 );
    setJmax( module, newCategoryIndex );
  }
  
  void learn( IModel &model,
              int id,
              NumericVector d ){
    List module = getModule( model.net, id );
    
    int nc = getNumCategories( module );
    if ( nc == 0 ){
      newCategory( model, module, d );
    }
    else{
      NumericVector a = activation( model, module, d );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      while( !resonance ){
        int J_max = T_j( j );
        double m = match( model, module, J_max, d );
        if ( m >= getRho( module ) ){
          setJmax( module, J_max );
          weightUpdate( model, module, J_max, d );
          counterUpdate( module, J_max );
          resonance = true;
          if ( hasMoreModules( model.net, id ) ){
            // match >= rho_a, then move up to the next module in the hierarchy
            // the weight of this node will be the input for the next module
            
            learn( model, id+1, d );
          }
        }
        else{
          if ( j == getNumCategories( module ) - 1 ){
            newCategory( model, module, d );
            resonance = true;
            if ( hasMoreModules( model.net, id ) ){
              // match >= rho_a, then move up to the next module in the hierarchy
              // the weight of this node will be the input for the next module
              // NumericVector w =  as<NumericMatrix>( module["w"] )( J_max, _ );
              learn( model, id+1, d );
            }
          }
          else{
            j++;
          } // if
        } // match
      } // while resonance
    } // if
    
  }
  
  int classify ( IModel &model,
                 int id,
                 NumericVector d ){
    List module = ART::getModule( model.net, id );
    int category = -1;
    
    NumericVector a = activation( model, module, d );
    NumericVector T_j = sortIndex( a );
    bool resonance = false;
    int j = 0;
    
    while(!resonance){
      int J_max = T_j( j );
      double m = match( model, module, J_max, d );
      if ( m >= getRho( module ) ){
        setJmax( module, J_max );
        category = J_max;
        resonance = true;
      }
      else{
        if ( j  == getNumCategories( module ) - 1 ){
          setJmax( module, category );
          resonance = true;
        }
        else{
          j++;
        } // if
      } // match
    } // while resonance
    
    return category;
  }
  
  void train( IModel &model,
              NumericMatrix x){
    
    int ep = getMaxEpochs( model.net );
    int nrow = x.rows();
    int numModules = getNumModules( model.net );
    for (int i = 1; i <= ep; i++){
      
      std::cout << "Epoch no. " << i << std::endl;
      
      int id = getModule( model.net, 0 )["id"];
      for ( int i = 0; i < nrow; i++ ){
        learn( model, id, model.processCode( x( i, _ ) ) );
      }
      
      for ( int j = 0; j < numModules; j++ ){
        int change = getModuleChange( ART::getModule( model.net, j ) );
        std::cout << "ID " << j << " Number of changes: " << change << std::endl;
      }
      if ( getTotalChange( model.net ) == 0 ) {
        ART::setEpoch( model.net, i );
        break;
      } else{
        if ( i < ep ){
          // only reset counters if it hasn't reached the maximum epoch
          // that way if the user wants to stop the learning using fewer epochs
          // then the node counters are still available for inspection
          int n  = getNumModules( model.net );
          for ( int i = 0; i < n; i++ ){
            List module = getModule( model.net, i );
            counterReset( module );
            changeReset( module );
          }
        }
      }
    }
    // subset weight matrix; loop through all modules
    int l = getNumModules( model.net );
    for ( int i = 0; i < l; i++ ){
      List module = getModule( model.net, i );
      int numCategories = getNumCategories( module );
      setWeightMatrix( module, subsetRows( getWeightMatrix( module ), numCategories ) );
      setCounterVector( module, subsetVector( getCounterVector( module ), numCategories ) );
      setChangeVector( module, subsetVector( getChangeVector( module ), numCategories ) );
    }
  }
  
  List predict( IModel &model,
                int id,
                NumericMatrix x ){
    List classified;
    int nrow = x.rows();
    NumericVector category( nrow );
    
    for (int i = 0; i < nrow; i++){
      // currently supports only one module
      int result = classify( model, id, model.processCode( x( i,_ ) ) );
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
void train ( List net, NumericMatrix x ){
  IModel *model;
  
  if ( isFuzzy( net ) ){
    model = new Fuzzy( net );
  }
  
  if ( isHypersphere( net ) ){
    model = new Hypersphere( net, x );
  }
  
  if ( !ART::isInitialized( net ) ) {
    ART::init( *model );
  }
  ART::train( *model, x );
  
  delete model;
  
}

// [[Rcpp::export(.predictART)]]
List predict ( List net, int id, NumericMatrix x ){
  IModel *model;
  
  if ( isFuzzy( net ) ){
    model = new Fuzzy( net );
  }

  if ( isHypersphere( net ) ){
    model = new Hypersphere( net );
  }

  
  List results = ART::predict( *model, id, x );
  delete model;
  return results;
}

// [[Rcpp::export(.ART)]]
List newART ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20 ){
  return ART::create( dimension, num, vigilance, learningRate, categorySize, maxEpochs );
}

bool isART ( List net ){
  return as<std::string>( net.attr( "class" ) ).compare( "ART" ) == 0;
}


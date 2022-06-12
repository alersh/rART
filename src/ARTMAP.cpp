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

namespace ARTMAP {

  // create and return a mapfield module
  List mapfield ( int id, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 50, bool simplified = false ){
    
    IntegerVector change;
    
    List module = List::create( _["id"] = id,                   // module id
                                _["weightDimension"] = 0      , // the number of dimensions in the weight
                                _["capacity"] = categorySize,   // number of category to create when the module runs out of categories to match
                                _["alpha"] = 0.001,             // activation function parameter
                                _["epsilon"] = 0.000001,        // match function parameter
                                _["rho"] = vigilance,           // vigilance parameter
                                _["beta"] = learningRate,       // learning parameter
                                _["change"] = change            // keep track of the number of changes in the mapfield
    );
    
    if ( simplified ){
      IntegerVector w;
      module.push_back( w, "w" );
      module.push_back( 0, "numCategories" );
    }
    else{
      NumericMatrix w;
      module.push_back( w, "w" );
      module.push_back( 0, "numCategories_a" );
      module.push_back( 0, "numCategories_b" );
    }
    
    return module;
  }
  
  List getMapfield ( List net ){
    return net["mapfield"];
  }
  
  List getModule_a( List net ){
    return as<List>( net["module"] )["a"];
  }
  
  List getModule_b( List net ){
    return as<List>( net["module"] )["b"];
  }

  namespace simplified {
  
    bool isSimplified( List net ){
      return as<bool>( net.attr( "simplified" ) );
    }
    
    int getWeight( List mapfield, int index ){
      return as<IntegerVector>( mapfield["w"] )[index];
    }
    
    void setWeight( List mapfield, int index, int w ){
      IntegerVector v = mapfield["w"];
      v( index ) = w;
    }
    
    IntegerVector getWeightVector( List mapfield ){
      return mapfield["w"];
    }
    
    void setWeightVector( List mapfield, IntegerVector v ){
      mapfield["w"] = v;
    }
    
    void newCategory( List mapfield, int label ){
      int numCategories = ART::getNumCategories( mapfield );
      int newCategoryIndex = numCategories;
      IntegerVector wm = getWeightVector( mapfield );
      if ( numCategories == wm.length() ){
        // reached the max capacity, so add more rows
        IntegerVector newWeight = appendVector( wm, ART::getCapacity( mapfield ) );
        setWeightVector( mapfield, newWeight );
        
        IntegerVector c = appendVector( ART::getChangeVector( mapfield ), ART::getCapacity( mapfield ) );
        ART::setChangeVector( mapfield, c );
      }
      setWeight( mapfield, newCategoryIndex, label );
      ART::incChange( mapfield, newCategoryIndex );
      ART::setNumCategories( mapfield, numCategories + 1 );
      
    }
    void initMapfield( List mapfield ){
      int size = ART::getCapacity( mapfield );
      
      IntegerVector w( size );
      setWeightVector( mapfield, w );
      
      IntegerVector c(size);
      ART::setChangeVector( mapfield, c );
    }
    
    void learn ( IModel &model, NumericVector d, int label){
      List module = ART::getModule( model.net, 0 );
      List mapfield = getMapfield( model.net );
      
      int nc = ART::getNumCategories( module );
      if ( nc == 0 ){
        
        ART::newCategory( model, module, d );
        newCategory( mapfield, label );
        
      }
      else{
        NumericVector a = ART::activation( model, module, d );
        NumericVector T_j = sortIndex( a );
        bool resonance = false;
        int j = 0;
        double rho = ART::getRho( module );
        while( !resonance ){
          int J_max = T_j( j );
          
          double m = ART::match( model, module, J_max, d );
          
          if ( m >= rho ){
            if ( getWeight( mapfield, J_max ) == label ){
              ART::setJmax( module, J_max );
              ART::weightUpdate( model, module, J_max, d );
              ART::counterUpdate( module, J_max );
              resonance = true;
            }
            else{
              rho = std::min( m + ART::getEpsilon( module ), 1.0 );
              
              if ( j == ART::getNumCategories( module ) - 1 ){
                ART::setJmax( module, j + 1 );
                ART::newCategory( model, module, d );
                newCategory( mapfield, label );
                resonance = true;
              }
              else{
                j++;
              } // if
            } // mapfield == label
          } // match >= rho_a
          else{
            if ( j == ART::getNumCategories( module ) - 1 ){
              ART::setJmax( module, j + 1 );
              ART::newCategory( model, module, d );
              newCategory( mapfield, label );
              resonance = true;
            }
            else{
              j++;
            } // if
          } // match fails
        } // while resonance
      } // if
      
    }
    
    int test( int predicted, int label ){
      int matched = NA_INTEGER;
      matched = predicted == label? 1 : 0;
      
      return matched;
    }
    
    // simplified classification
    List classify( IModel &model, NumericVector d ){
      
      List module = getModule_a( model.net );
      List mapfield = getMapfield( model.net );
      int category = NA_INTEGER;
      int predicted = NA_INTEGER;
      
      int nc = ART::getNumCategories( module );
      
      NumericVector a = ART::activation( model, module, d );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho = ART::getRho( module );
      while( !resonance ){
        int J_max = T_j( j );
        
        double m = ART::match( model, module, J_max, d );
        
        if ( m >= rho ){
          
          category = J_max;
          predicted = getWeight( mapfield, J_max );
          resonance = true;
          
        } // match >= rho_a
        else{
          if ( j == nc - 1 ){
            // can't find a match
            ART::setJmax( module, NA_INTEGER );
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
  
    void subsetMapfield( List mapfield ){
      int numCategories = ART::getNumCategories( mapfield );
      setWeightVector( mapfield, subsetVector( getWeightVector( mapfield ), numCategories ) );
      ART::setChangeVector( mapfield, subsetVector( ART::getChangeVector( mapfield ), numCategories ) );
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
        F1( ART::getWeightMatrix( module_b ).cols() );
        // Can't find the node b in F2. Something is wrong. Return all NA
        F1.fill( NA_INTEGER );
        return F1;
      }
      F1 = ART::getWeight( module_b, nodeIndex_b ); 
      
      return F1;
    }
    
    double match( IModel &model, List mapfield, int nodeIndex_a, int nodeIndex_b ){
      NumericVector w_a = ART::getWeight( mapfield, nodeIndex_a );
      NumericVector yb( w_a.length(), 0 );
      yb( nodeIndex_b ) = 1;
      
      double a = model.match( mapfield, yb, w_a );
      return a;
    }
    
    void mapfieldUpdate( IModel &model, List mapfield, int nodeIndex_a, int nodeIndex_b ){
      NumericVector w_a = ART::getWeight( mapfield, nodeIndex_a );
      NumericVector y_b( w_a.length(), 0 );
      y_b( nodeIndex_b ) = 1;
      ART::weightUpdate( model, mapfield, nodeIndex_a, y_b );
      
    }
    
    void setNumCategories_a( List mapfield, int x ){
      mapfield["numCategories_a"] = x;
    }
    
    void setNumCategories_b( List mapfield, int x ){
      mapfield["numCategories_b"] = x;
    }
    
    int getNumCategories_a( List mapfield ){
      return mapfield["numCategories_a"];
    }
    
    int getNumCategories_b( List mapfield ){
      return mapfield["numCategories_b"];
    }
    
    List getModule_a( List net ){
      return as<List>( net["module"] )["a"];
    }
    
    List getModule_b( List net ){
      return as<List>( net["module"] )["b"];
    }
    
    List getMapfield( List net ){
      return net["mapfield"];
    }
  
    void newCategory_a( List mapfield ){
      
      int numCategories = getNumCategories_a( mapfield );
      int newCategoryIndex = numCategories;
      NumericMatrix wm = ART::getWeightMatrix( mapfield );
      NumericMatrix newWeight;
      if ( numCategories == wm.rows() ){
        // reached the max capacity, so add more rows
        newWeight = appendRows( wm, ART::getCapacity( mapfield ) );
        ART::setWeightMatrix( mapfield, newWeight );
        
        IntegerVector c = appendVector( ART::getChangeVector( mapfield ), ART::getCapacity( mapfield ) );
        ART::setChangeVector( mapfield, c );
      }
      else{
        newWeight = wm;
      }
      
      ART::setWeight( mapfield, newCategoryIndex, rep( 1.0, wm.cols() ) );
      ART::incChange( mapfield, newCategoryIndex );
      setNumCategories_a( mapfield, numCategories + 1 );
      
    }
    
    void newCategory_b( List mapfield ){
      
      int numCategories = getNumCategories_b( mapfield );
      int newCategoryIndex = numCategories;
      NumericMatrix wm = ART::getWeightMatrix( mapfield );
      NumericMatrix newWeight;
      if ( numCategories == wm.cols() ){
        // reached the max capacity, so add more columns
        newWeight = appendColumns( wm, ART::getCapacity( mapfield ) );
      }
      else{
        newWeight = wm;
      }
      newWeight( wm.rows() - 1, newCategoryIndex ) = 0;
      ART::setWeightMatrix( mapfield, newWeight );
      ART::setWeightDimension( mapfield, numCategories + 1 );
      setNumCategories_b( mapfield, numCategories + 1 );
    }
  
    void initMapfield( List mapfield ){
      int size = ART::getCapacity( mapfield );
      
      NumericMatrix w( 1, size );
      ART::setWeightMatrix( mapfield, w );
      
      IntegerVector c( size );
      ART::setChangeVector( mapfield, c );
    }
  
    void learn ( IModel &model, NumericVector d, NumericVector label ){
   
      List module_a = getModule_a( model.net );
      List module_b = getModule_b( model.net );
      List mapfield = getMapfield( model.net );
      
      // init
      int nc_a = ART::getNumCategories( module_a );
      int nc_b = ART::getNumCategories( module_b );
      int nc_ab_a = getNumCategories_a( mapfield );
      int nc_ab_b = getNumCategories_b( mapfield );
      if ( nc_a == 0  && nc_b == 0  && nc_ab_a == 0 && nc_ab_b == 0 ){
        
        ART::newCategory( model, module_a, d );
        ART::newCategory( model, module_b, label );
        // Add new category in b first before a
        newCategory_b( mapfield );
        newCategory_a( mapfield );
        mapfieldUpdate( model, mapfield, 0, 0 );
        
      }
      else{
        ART::learn( model, ART::getID( module_b ), label );
        // add a new ab category whenever a new category is added in F2b
        if ( ART::getNumCategories( module_b ) > getNumCategories_b( mapfield ) ){
          // update nodes in mapfield
          newCategory_b( mapfield );
        }
        // get ART a F2 activations
        NumericVector a = ART::activation( model, module_a, d );
        NumericVector T_j = sortIndex( a );
        bool resonance = false;
        int j = 0;
        double rho_a = ART::getRho( module_a );
        while( !resonance ){
          int Jmax_a = T_j( j );
          
          double m = ART::match( model, module_a, Jmax_a, d );
          
          if ( m >= rho_a ){
            // check the mapfield
            
            double m_ab = match( model, mapfield, Jmax_a, ART::getJmax( module_b ) );
            
            if ( m_ab >= ART::getRho( mapfield ) ){
              ART::setJmax( module_a, Jmax_a );
              ART::weightUpdate( model, module_a, Jmax_a, d );
              ART::counterUpdate( module_a, Jmax_a );
              mapfieldUpdate( model, mapfield, Jmax_a, ART::getJmax( module_b ) );
              
              resonance = true;
            }
            
            if ( !resonance ){
              
              rho_a = std::min( m + ART::getEpsilon( module_a ), 1.0 );
              
              // if run out of categories, then add a new one
              if ( j == ART::getNumCategories( module_a ) - 1 ){
                j++;
                ART::setJmax( module_a, j );
                
                // add a new F2 node in ART a
                ART::newCategory( model, module_a, d );
                
                // add a new node in ART ab
                newCategory_a( mapfield );
                mapfieldUpdate( model, mapfield, j, ART::getJmax( module_b ) );
                
                resonance = true;
              }
              else{
                j++;
              } // if
            } // resonance
          }
          else{
            if ( j == ART::getNumCategories( module_a ) - 1 ){
              j++;
              ART::setJmax( module_a, j );
              ART::newCategory( model, module_a, d );
              
              // add a new node in ART ab
              newCategory_a( mapfield );
              
              mapfieldUpdate( model, mapfield, j, ART::getJmax( module_b ) );
              
              resonance = true;
              
            }
            else{
              j++;
            } // if
          } // match fails
        } // while resonance
      } // if
      
    }
    
    int test( IModel &model, NumericVector label ) {
      
      int matched = NA_INTEGER;
      
      List mapfield = getMapfield( model.net );
      List module_a = getModule_a( model.net );
      List module_b = getModule_b( model.net );
      
      int category_b = ART::classify( model, ART::getID( module_b ), label );
      
      int Jmax_a = ART::getJmax( module_a );
      
      if ( category_b != NA_INTEGER && Jmax_a != NA_INTEGER ){
        double m_ab = match( model, mapfield, Jmax_a, category_b );
        matched = m_ab >= ART::getRho( mapfield ) ? 1 : 0;
      }
      
      return matched;
    }
    
    List classify( IModel &model, NumericVector d ){
      
      List mapfield = getMapfield( model.net );
      List module_a = getModule_a( model.net );
      List module_b = getModule_b( model.net );
      
      // best matching node in F2a
      int category_a = NA_INTEGER;
      
      NumericVector predicted( getNumCategories_b( mapfield ), NA_REAL );
      NumericVector F1_b( ART::getWeightMatrix( module_b ).cols(), NA_REAL );
      
      NumericVector a = ART::activation( model, module_a, d );
      NumericVector T_j = sortIndex( a );
      bool resonance = false;
      int j = 0;
      double rho_a = ART::getRho( module_a );
      while( !resonance ){
        int Jmax_a = T_j( j );
        double m = ART::match( model, module_a, Jmax_a, d );
        
        if ( m >= rho_a ){
          // prediction
          category_a = Jmax_a;
          ART::setJmax( module_a, Jmax_a );
          predicted = ART::getWeight( mapfield, Jmax_a );
          F1_b = recall( module_b, predicted );
          
          resonance = true;
        }
        else {
          // raise the vigilance of a and continue with the search
          rho_a = std::min( rho_a + ART::getEpsilon( module_a ), 1.0 );
          
          // if it runs out of categories, then it can't find a match
          if ( j == ART::getNumCategories( module_a ) - 1 ){
            ART::setJmax( module_a, NA_INTEGER );
            resonance = true;
          }
          else{
            // search for the next best matching node in F2a
            j++;
          } // if
        } // else
        
      } // while resonance
      
      return List::create( _["category_a"] = category_a,
                           _["F1_b"] = F1_b );
      
    }
  
    void subsetMapfield( List mapfield ){
      int numCategories_a = getNumCategories_a( mapfield );
      int numCategories_b = getNumCategories_b( mapfield );
      ART::setWeightMatrix( mapfield, subsetRows( subsetColumns( 
                                                                ART::getWeightMatrix( mapfield ), numCategories_b 
                                                               ),
                                                  numCategories_a )
                          );
      ART::setChangeVector( mapfield, subsetVector( ART::getChangeVector( mapfield ), numCategories_a ) );
    }
  
  }

  void init( IModel &model ){
    List mapfield = getMapfield( model.net );
    if ( simplified::isSimplified( model.net ) ){
      simplified::initMapfield( mapfield );
    } else{
      standard::initMapfield( mapfield );
    }
  }

  void train( IModel &model,
              NumericMatrix x,
              Nullable<NumericVector> vTarget,
              Nullable< NumericMatrix > mTarget ){
    
    int ep = ART::getMaxEpochs( model.net );
    int nrow = x.rows();
    if ( !isARTMAP( model.net ) ){
      stop( "The network is not an ARTMAP." );
    }
    if ( !mTarget.isNotNull() && !simplified::isSimplified( model.net ) ){
      stop( "The labels are missing. End running." );
    }
    else if ( !vTarget.isNotNull() && simplified::isSimplified( model.net ) ){
      stop("The labels are missing. End running.");
    }
    for (int i = 1; i <= ep; i++){
      std::cout << "Epoch no. " << i << std::endl;
      
      for (int j = 0; j < nrow; j++){
        if ( simplified::isSimplified( model.net ) )
          simplified::learn( model, model.processCode( x( j, _ ) ), NumericVector ( vTarget )( j ) );
        else 
          standard::learn( model, model.processCode( x( j, _ ) ), model.processCode( NumericMatrix ( mTarget )( j, _ ) ) );
      }
      
      List mapfield = getMapfield( model.net );
      int change = ART::getTotalChange( model.net ) + ART::getModuleChange( mapfield );
      cout << "Number of changes " << change << endl;
      if ( change == 0 ) {
        ART::setEpoch( model.net, i );
        break;
      } else{
        List module_a = getModule_a( model.net );
        ART::changeReset( module_a );
        ART::counterReset( module_a );
        
        ART::changeReset( mapfield );
        
        if ( !simplified::isSimplified( model.net ) ){
          List module_b = getModule_b( model.net );
          ART::changeReset( module_b );
          ART::counterReset( module_b );
        }
      }
    }
    // subset module weight matrix, counter and change vectors
    int l = ART::getNumModules( model.net );
    for ( int i = 0; i < l; i++ ){
      List module = ART::getModule( model.net, i );
      int numCategories = ART::getNumCategories( module );
      ART::setWeightMatrix( module, subsetRows( ART::getWeightMatrix( module ), numCategories ) );
      ART::setCounterVector( module, subsetVector( ART::getCounterVector( module ), numCategories ) );
      ART::setChangeVector( module, subsetVector( ART::getChangeVector( module ), numCategories ) );
    }
    
    // subset mapfield weight matrix, counter and change vectors
    if ( simplified::isSimplified( model.net ) ){
      simplified::subsetMapfield( getMapfield( model.net ) );
    }
    else{
      standard::subsetMapfield( getMapfield( model.net ) );
    }
  }
  
  List predict( IModel &model,
                NumericMatrix x,
                Nullable< NumericVector > vTarget = R_NilValue ,
                Nullable< NumericMatrix > mTarget = R_NilValue){
    List classified;
    int nrow = x.rows();
    int ncol = x.cols();
    IntegerVector category_a( nrow ), matched( nrow );
    
    if ( simplified::isSimplified( model.net ) ){
      NumericVector predicted ( nrow );
      for ( int i = 0; i < nrow; i++ ){
        
        List result = simplified::classify( model, model.processCode( x( i,_ ) ) );
        category_a( i ) = result( "category_a" );
        predicted( i ) = result( "predicted" );
        if ( vTarget.isNotNull() ){
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
        List result = standard::classify( model, model.processCode( x( i,_ ) ) );
        predicted( i, _ ) = model.unProcessCode( result( "F1_b" ) );
        category_a( i ) = result( "category_a" );
        if ( mTarget.isNotNull() ){
          matched( i ) = standard::test( model, model.processCode( NumericMatrix ( mTarget )( i, _ ) ) );
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
List newARTMAP ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20, bool simplified = false ){
  if ( vigilance < 0.0 || vigilance > 1.0 ){
    stop( "The vigilance value must be between 0 and 1.0." );
  }
  
  if ( learningRate < 0.0 || learningRate > 1.0 ){
    stop( "The learningRate value must be between 0 and 1.0." );
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
  
  List net;
  if ( !simplified ){
    net = newART( dimension, 2, vigilance, learningRate, categorySize, maxEpochs );
    as<List>( net["module"] ).attr( "names" ) = CharacterVector::create( "a", "b" );
  }
  else{
    net = newART( dimension, 1, vigilance, learningRate, categorySize, maxEpochs );
    as<List>( net["module"] ).attr( "names" ) = CharacterVector::create( "a" );
  }
  //mapfield.push_back( ARTMAP::mapfield( 0, vigilance, learningRate, simplified ), "ab" );
  
  List mapfield = ARTMAP::mapfield( 0, vigilance = vigilance, learningRate = learningRate, categorySize = 50, simplified = simplified );
  net.push_back( mapfield, "mapfield" );
  net.attr( "class" ) = "ARTMAP";
  net.attr( "simplified" ) = simplified;
  return net;
}

// [[Rcpp::export(.trainARTMAP)]]
void trainARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue ){
  IModel *model;
  
  if ( isFuzzy( net ) ){
    model = new Fuzzy( net );
  }
  
  if ( isHypersphere( net ) ){
    model = new Hypersphere( net, x );
  }
  
  ART::init( *model );
  ARTMAP::train( *model, x, vTarget, mTarget );
  
  delete model;
  
}

// [[Rcpp::export(.predictARTMAP)]]
List predictARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue ){
  
  IModel *model;
  if ( isFuzzy( net ) ){
    model = new Fuzzy ( net );
  }
  if ( isHypersphere( net ) ){
    model = new Hypersphere( net, x );
  }
  
  List results = ARTMAP::predict( *model, x, vTarget, mTarget );
  delete model;
  return results;
}
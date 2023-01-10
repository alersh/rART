/****************************************************************************
 *
 *  ARTMAP.h
 *  ARTMAP
 *
 ****************************************************************************/


#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;

#ifndef ARTMAP_H
#define ARTMAP_H

List newARTMAP ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20, bool simplified = false );
bool isARTMAP ( List net );

namespace ARTMAP{
  List mapfield ( int id, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 50, bool simplified = false );
  namespace simplified{
  bool isSimplified( List net );
    void newCategory( List mapfield, int label );
    int learn ( IModel &model,
                NumericVector d,
                int label );
    List classify( IModel &model,
                   NumericVector d,
                   bool test = false );
    int test( int predicted, int label );
  
  }

  namespace standard{
  
    int mapfieldUpdate( IModel &model, List mapfield, int nodeIndex_a, int nodeIndex_b );
    
    int learn ( IModel &model,
                NumericVector d,
                NumericVector label );
    int test( IModel &model, 
              NumericVector label ) ;
    
    List classify( IModel &model,
                   NumericVector d,
                   bool test = false );
  }

  void train( IModel &model,
              NumericMatrix x,
              Nullable< NumericVector > vTarget,
              Nullable< NumericMatrix > mTarget );
  
  List predict( IModel &model,
                NumericMatrix x,
                Nullable< NumericVector > vTarget,
                Nullable< NumericMatrix > mTarget,
                bool test = false );
}

void trainARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue );

List predictARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false );

#endif

/****************************************************************************
 * 
 *  topoART.h
 *  TopoART and TopoARTMAP
 * 
 *  by Albert Ler
 *  © 2021
 *  
 ****************************************************************************/

#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;
using namespace std;

#ifndef TOPOART_H
#define TOPOART_H

bool isTopoART ( List net );

namespace Topo{

List module( int id, int dimension,double vigilance, int phi, double learningRate1, double learningRate2, int categorySize = 200 );

double rho ( double rho, int moduleId );

void train ( IModel &model, NumericMatrix x );
NumericVector classify ( IModel &model,
                         int id,
                         NumericVector d );
List predict( IModel &model,
              int id,
              NumericMatrix x );

}


List TopoART ( int dimension, int num = 2, double vigilance = 0.9, double learningRate1 = 1.0, double learningRate2 = 0.6, int tau = 100, int phi = 6, int categorySize = 200, int maxEpochs = 20 );
void topoTrain( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue );
List topoPredict(  List net, int id, NumericMatrix x );

#endif

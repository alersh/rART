/****************************************************************************
 *
 *  topoART.h
 *  TopoART and TopoARTMAP
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;
using namespace std;


bool isTopoART ( List net );

namespace Topo{

  List module( int id, int numFeatures, double vigilance, int phi, double learningRate1, double learningRate2, int categorySize = 200 );
  void learn ( List net,
               int id,
               NumericVector d,
               NumericVector change,
               std::function< double( List, NumericVector, NumericVector ) > activationFun,
               std::function< double( List, NumericVector, NumericVector ) > matchFun,
               std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun );

  void train( List net,
              NumericMatrix x,
              std::function< NumericVector( NumericVector ) > codeFun,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun );

  NumericVector classify( List net,
                          int id,
                          NumericVector d,
                          std::function< double( List, NumericVector, NumericVector ) > activationFun,
                          std::function< double( List, NumericVector, NumericVector ) > matchFun,
                          NumericVector classified );
  List predict ( List net,
                 int id,
                 NumericMatrix x,
                 std::function< NumericVector( NumericVector ) > codeFun,
                 std::function< double( List, NumericVector, NumericVector ) > activationFun,
                 std::function< double( List, NumericVector, NumericVector ) > matchFun );

}


List TopoART ( int numFeatures, int num = 2, double vigilance = 0.9, int tau = 100, int phi = 6, double learningRate1 = 1.0, double learningRate2 = 0.6, int categorySize = 200, int maxEpochs = 20 );
void topoTrain( List net, NumericMatrix x, Nullable< NumericVector > labels = R_NilValue );
List topoPredict(  List net, int id, NumericMatrix x );



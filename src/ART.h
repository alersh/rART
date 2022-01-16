/****************************************************************************
 *
 *  ART.h
 *  ART
 *
 ****************************************************************************/


#include <Rcpp.h>
using namespace Rcpp;
using namespace std;


List newART ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 200, int maxEpochs = 20 );
bool isART ( List net );
namespace ART {
        List module ( int id, int dimension, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100 );
        NumericVector activation( List module, NumericVector x, std::function< double( List, NumericVector, NumericVector ) > fun );
        double match( List module, NumericVector x, int weightIndex, std::function< double( List, NumericVector, NumericVector ) > fun );
        int weightUpdate( List module, NumericVector x, int weightIndex, std::function< NumericVector( List, NumericVector, NumericVector ) > fun );
        void newCategory( List net, NumericVector x );
        int learn( List net,
                   int id,
                   NumericVector d,
                   std::function< double( List, NumericVector, NumericVector ) > activationFun,
                   std::function< double( List, NumericVector, NumericVector ) > matchFun,
                   std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun );
        int classify( List net,
                      int id,
                      NumericVector d,
                      std::function< double( List, NumericVector, NumericVector ) > activationFun,
                      std::function< double( List, NumericVector, NumericVector ) > matchFun );

        void train( List net,
                    NumericMatrix x,
                    std::function< NumericVector( NumericVector ) > codeFun,
                    std::function< double( List, NumericVector, NumericVector ) > activationFun,
                    std::function< double( List, NumericVector, NumericVector ) > matchFun,
                    std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun );
        List predict( List net,
                      int id,
                      NumericMatrix x,
                      std::function< NumericVector( NumericVector ) > codeFun,
                      std::function< double( List, NumericVector, NumericVector ) > activationFun,
                      std::function< double( List, NumericVector, NumericVector ) > matchFun );
}

void trainART ( List net, NumericMatrix x );

List predictART ( List net, NumericMatrix x );



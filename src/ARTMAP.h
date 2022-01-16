/****************************************************************************
 *
 *  ARTMAP.h
 *  ARTMAP
 *
 ****************************************************************************/


#include <Rcpp.h>
using namespace Rcpp;
using namespace std;


List newARTMAP ( int dimension, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 200, int maxEpochs = 20, bool simplified = false );
bool isARTMAP ( List net );
bool isSimplified( List net );

namespace ARTMAP{

  namespace simplified{

    void newCategory( List net, int label );
    int learn ( List net,
                NumericVector d,
                int label,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun );
    List classify( List net,
                   NumericVector d,
                   std::function< double( List, NumericVector, NumericVector ) > activationFun,
                   std::function< double( List, NumericVector, NumericVector ) > matchFun,
                   bool test );
    int test( int predicted, int label );

  }

  namespace standard{

    int mapfieldUpdate( List net, int nodeIndex_a, int nodeIndex_b, std::function< NumericVector( List, NumericVector, NumericVector ) > fun );

    int learn ( List net,
                NumericVector d,
                NumericVector label,
                std::function< double( List, NumericVector, NumericVector ) > activationFun,
                std::function< double( List, NumericVector, NumericVector ) > matchFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun,
                std::function< NumericVector( List, NumericVector, NumericVector ) > mapfieldUpdateFun );
    int test( List net,
              NumericVector label,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun ) ;

    List classify( List net,
                   NumericVector d,
                   std::function< double( List, NumericVector, NumericVector ) > activationFun,
                   std::function< double( List, NumericVector, NumericVector ) > matchFun,
                   bool test = false );
  }

  void train( List net,
              NumericMatrix x,
              Nullable< NumericVector > vTarget,
              Nullable< NumericMatrix > mTarget,
              std::function< NumericVector( NumericVector ) > codeFun,
              std::function< double( List, NumericVector, NumericVector ) > activationFun,
              std::function< double( List, NumericVector, NumericVector ) > matchFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > weightUpdateFun,
              std::function< NumericVector( List, NumericVector, NumericVector ) > mapfieldUpdateFun );

 List predict( List net,
               NumericMatrix x,
               Nullable< NumericVector > vTarget,
               Nullable< NumericMatrix > mTarget,
               std::function< NumericVector( NumericVector ) > codeFun,
               std::function< NumericVector( NumericVector ) > uncodeFun,
               std::function< double( List, NumericVector, NumericVector ) > activationFun,
               std::function< double( List, NumericVector, NumericVector ) > matchFun,
               bool test = false );
}
void trainARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue );

List predictARTMAP ( List net, NumericMatrix x, Nullable< NumericVector > vTarget = R_NilValue, Nullable< NumericMatrix > mTarget = R_NilValue, bool test = false );



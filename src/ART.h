/****************************************************************************
 *
 *  ART.h
 *  ART
 *
 ****************************************************************************/


#include <Rcpp.h>
#include "IModel.h"
using namespace Rcpp;
using namespace std;

#ifndef ART_H
#define ART_H


List newART ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 200, int maxEpochs = 20 );
bool isART ( List net );

namespace ART {
        List create ( int dimension, int num = 1, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100, int maxEpochs = 20 );
        int getID( List module );
        int getJmax( List module, int matchIndex = 0 );
        void setJmax( List module, int J, int matchIndex = 0 );
        void addJmax( List module, int J );
        int getMaxEpochs( List net );
        void setEpoch( List net, int epoch );
        void incChange( List module, int index );
        int getChangeSum( List module );
        IntegerVector getChangeVector( List module );
        void setChangeVector( List module, IntegerVector v );
        void changeReset( List module );
        bool unchanged( List module );
        int getNumModules( List net );
        List module ( int id, double vigilance = 0.75, double learningRate = 1.0, int categorySize = 100 );
        List getModule( List net, int moduleID );
        void setModule( List net, List module );
        bool hasMoreModules( List net, int currentModuleID );
        NumericMatrix getWeightMatrix( List module );
        void setWeightMatrix( List module, NumericMatrix w );
        void initWeights( List module, int dim );
        NumericVector getWeight( List module, int weightIndex );
        void setWeight( List module, int weightIndex, NumericVector w );
        double getRho( List module );
        void setRho( List module, double rho );
        int getNumCategories( List module );
        void setNumCategories( List module, int x );
        double getLearningRate( List module );
        void setLearningRate( List module, double learningRate );
        int getDimension( List module );
        int getCapacity( List module );
        int getCounter( List module, int weightIndex );
        IntegerVector getCounterVector( List module );
        double getAlpha( List module );
        double getEpsilon( List module );
        void setCounterVector( List module, IntegerVector v );
        void counterReset( List module );
        void initModule( List module, int weightDimension );
        void init( IModel &model );
        
        NumericVector activation( IModel &model, List module, NumericVector x );
        double match( IModel &model, List module, int weightIndex , NumericVector x);
        void weightUpdate( IModel &model, List module, int weightIndex, NumericVector x );
        void counterUpdate( List module, int nodeIndex );
        void newCategory( IModel &model, List module, NumericVector x );
        void learn( IModel &model, int id, NumericVector d );
        int classify( IModel &model, int id, NumericVector d );
        
        void train( IModel &model, NumericMatrix x );
        List predict( IModel &model, int id, NumericMatrix x );
}

void train ( List net, NumericMatrix x );
List predict ( List net, int id, NumericMatrix x );

#endif

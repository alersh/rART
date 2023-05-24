#include <Rcpp.h>
using namespace Rcpp;

#ifndef IMODEL_H
#define IMODEL_H

struct IModel{
  List net;
  IModel ( List net ){
    this->net = net;
  };  
  virtual ~IModel(){};
  
  /* getWeightDimension: Return the total dimension of the weight vector */
  virtual int getWeightDimension( int featureDimension ) = 0;
  
  /* newWeight:  additional set up for the new weight vector. For example, for the hypersphere model
     the weight vector needs to append a radius element to the end of the new weight vector. This can 
     be done here. */
  virtual NumericVector newWeight( NumericVector w ) = 0;
  
  /* activation: Calculate the activation values */
  virtual double activation( List module, NumericVector x, NumericVector w ) = 0;
  
  /* match: Calculate the match values between the input vector x and the weight vector w */
  virtual double match( List module, NumericVector x, NumericVector w ) = 0;
  
  /* weightUpdate: Update the weight vector */
  virtual NumericVector weightUpdate( List module, double learningRate, NumericVector x, NumericVector w ) = 0;
  
  /* processCode: The processing of the input code specific for this model. For example,
     the fuzzy model requires code complement and it can be done here. */
  virtual NumericVector processCode( NumericVector x ) { return x; };
  
  /* unProcessCode: Revert the processed code back to its original code. */
  virtual NumericVector unProcessCode( NumericVector x ) { return x; };
};

#endif
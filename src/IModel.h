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
  
  virtual int getWeightDimension( int featureDimension ) = 0;
  virtual NumericVector newWeight( NumericVector x ) = 0;
  virtual double activation( List module, NumericVector x, NumericVector w ) = 0;
  virtual double match( List module, NumericVector x, NumericVector w ) = 0;
  virtual NumericVector weightUpdate( List module, double learningRate, NumericVector x, NumericVector w ) = 0;
  virtual NumericVector processCode( NumericVector x ) { return x; };
  virtual NumericVector unProcessCode( NumericVector x ) { return x; };
};

#endif
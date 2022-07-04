#include <testthat.h>
#include "fuzzy.h"
#include "utils.h"

context("fuzzy") {
  test_that("activation") {
    Fuzzy *f = new Fuzzy( List::create( 0 ) );
    List module = List::create( _["alpha"] = 2 );
    NumericVector x = NumericVector::create( 1,2,3 );
    NumericVector w = NumericVector::create( 3,2,1 );
    double a = f->activation( module, x, w );
    expect_true( a == 0.5 );
    
    x = NumericVector::create(1,2,NA_REAL);
    a = f->activation( module, x , w );
    expect_true( a == 0.375 );
    
    x = NumericVector::create( 1,2,3 );
    w = NumericVector::create( NA_REAL,2,1 );
    a = f->activation( module, x, w );
    expect_true( a == 0.6 );
  }
  
  test_that("match") {
    Fuzzy *f = new Fuzzy( List::create( 0 ) );
    List module = List::create( _["alpha"] = 2 );
    NumericVector x = NumericVector::create( 1,2,3 );
    NumericVector w = NumericVector::create( 3,2,0 );
    double m = f->match( module, x, w );
    expect_true( m == 0.5 );
    
    x = NumericVector::create( NA_REAL,2,3 );
    m = f->match( module, x, w );
    expect_true( m == 0.4 );
  }
  
  test_that("weightUpdate"){
    Fuzzy *f = new Fuzzy( List::create( 0 ) );
    List module = List::create( _["alpha"] = 2 );
    NumericVector x = NumericVector::create( 1,2,3 );
    NumericVector w = NumericVector::create( 3,2,0 );
    double learningRate = 1.0;
    NumericVector z = f->weightUpdate( module, learningRate, x, w );
    NumericVector actual = NumericVector::create( 1,2,0 );
    expect_true( equal( z, actual ) );
  }
}



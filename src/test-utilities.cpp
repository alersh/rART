#include <testthat.h>
#include "utils.h"

context("utilities") {

  test_that("sortIndex") {
    NumericVector v = NumericVector::create(3,2,5,6,3,4);
    NumericVector sortedv = NumericVector::create(3,2,5,0,4,1);
    NumericVector test = sortIndex(v);
    int l = v.length();
    for (int i = 0; i < l; i++){
      expect_true(test[i] == sortedv[i]);
    }
  }

  NumericMatrix m(5, 3);
  m(_, 0) = NumericVector::create(2,4,1,2,5);
  m(_, 1) = NumericVector::create(7,5,3,5,4);
  m(_, 2) = NumericVector::create(1,2,5,3,0);

  test_that("colMax"){
    NumericVector maxes = colMax(m);
    expect_true(maxes[0] == max(m(_, 0)));
    expect_true(maxes[1] == max(m(_, 1)));
    expect_true(maxes[2] == max(m(_, 2)));
  }

  test_that("colMin"){
    NumericVector mins = colMin(m);
    expect_true(mins[0] == min(m(_, 0)));
    expect_true(mins[1] == min(m(_, 1)));
    expect_true(mins[2] == min(m(_, 2)));
  }

  test_that("appendRow"){
    NumericMatrix w = appendRows(m, 3);
    expect_true(w.rows() == m.rows() + 3);
    expect_true(w.cols() == m.cols());
  }

  test_that("joinVectors"){
    NumericVector v1 = NumericVector::create(2,3,1);
    NumericVector v2 = NumericVector::create(1,5);
    NumericVector v = joinVectors(v1, v2);
    NumericVector toMatch = NumericVector::create(2,3,1,1,5);
    int l = v.length();
    for (int i = 0; i < l; i++){
      expect_true(v[i] == toMatch[i]);
    }
  }
  
  test_that("createDummyCodeMap"){
    List code = List::create( _["1"] = IntegerVector::create(1,0,0),
                              _["2"] = IntegerVector::create(0,1,0),
                              _["3"] = IntegerVector::create(0,0,1) );
    StringVector v = StringVector::create("1","2","3");
    List test = createDummyCodeMap( v );
    expect_true( test.length() == code.length() );
    for ( int i = 0; i < code.length(); i++ ){
      IntegerVector c = code[i];
      IntegerVector t = test[i];
      expect_true( c.length() == t.length() );
      for ( int j = 0; j < c.length(); j++ ){
        expect_true( c[j] == t[j] );
      }
      
    }
  }

  test_that("encodeNumericLabel") {
    IntegerVector v = IntegerVector::create(1,2,3,2,3,2);
    List code = createDummyCodeMap( StringVector::create( "1", "2", "3"));
    IntegerVector actual = IntegerVector::create(1,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0);
    actual.attr("dim") = Dimension(6, 3);
    NumericMatrix actualm = as<NumericMatrix>(actual);
    NumericMatrix converted = encodeNumericLabel( v, code );
    int r = converted.rows();
    int c = converted.cols();
    for ( int i = 0; i <  r; i++ ){
      for ( int j = 0; j <  c; j++ ){
        expect_true( converted(i, j) == actualm(i, j) );
      }
    }
    
  }

}



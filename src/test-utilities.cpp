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

  test_that("appendVector"){
    NumericVector v1 = NumericVector::create(2,3,1);
    NumericVector v2 = NumericVector::create(1,5);
    NumericVector v = appendVector(v1, v2);
    NumericVector toMatch = NumericVector::create(2,3,1,1,5);
    int l = v.length();
    for (int i = 0; i < l; i++){
      expect_true(v[i] == toMatch[i]);
    }
  }

}



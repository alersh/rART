/****************************************************************************
 *
 *  utils.cpp
 *  ART utilities
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;
using namespace std;


// sortIndex: Sort the values in descending order and return the indices of the values
NumericVector sortIndex( NumericVector x ){
  NumericVector idx( x.size() );
  iota( idx.begin(), idx.end(), 0 );

  stable_sort( idx.begin(), idx.end(), [&x]( size_t i1, size_t i2 ) { return x[i1] > x[i2]; } );

  return idx;
}

NumericVector colMax( NumericMatrix x ){
  int cols = x.cols();
  NumericVector v( cols );
  for ( int i = 0; i < cols; i++ ){
    v( i ) = max( na_omit( x( _, i ) ) );
  }
  return v;
}

NumericVector colMin( NumericMatrix x ){
  int cols = x.cols();
  NumericVector v( cols );
  for ( int i = 0; i < cols; i++ ){
    v( i ) = min( na_omit( x( _, i ) ) );
  }
  return v;
}


// sameCode: A function that returns the vector x as is.
NumericVector sameCode( NumericVector x ){
  return x;
}

// complementCode: Create complement code
NumericVector complementCode( NumericVector x ){
  int s = x.size();
  NumericVector c( s*2 );
  for ( int i = 0; i < s; i++ ){
    c[i] = x[i];
    c[i+s] = 1 - x[i];
  }
  return c;
}

// uncomplementCode: Remove the complement part of the code and return the original
NumericVector uncomplementCode( NumericVector x ){
  int l = x.length();
  // the length must be even
  if ( l % 2 != 0 )
    stop( "The length of the code must be an even number." );

  return x[ Range( 0, l/2 ) ];
}

// appendRows: append rows to a matrix
NumericMatrix appendRows( NumericMatrix x, int numRows ){
  int nrow = x.nrow();
  int ncol = x.ncol();
  NumericMatrix v ( nrow + numRows, ncol );
  for (int i = 0; i < nrow; i++){
    v ( i,_ ) = x( i,_ );

  }
  return v;
}

// appendVector: append a vector
NumericVector appendVector( NumericVector v1, NumericVector v2 ){
  int l = v2.length();
  for ( int i = 0; i < l; i++ ){
    v1.push_back( v2[i] );
  }
  return v1;
}

// initComplementWeight: initialize weight and its complement
void initWeight( List net, bool complement = false ){
  // initialize weight dimension
  int dim = as<int>( net["numFeatures"] );
  if ( complement ){
    dim *= 2;
  }

  int size = as<int>( net["capacity"] );
  NumericMatrix w = net["w"];
  w = no_init( size, dim );
  net["w"] = w;
}


NumericMatrix subsetRows( NumericMatrix x, int rows ){

  if (rows > x.rows()){
    stop( "Too many rows have been selected." );
  }

  NumericMatrix y = x( Range( 0, rows - 1 ), _ );
  return y;
}


NumericVector vectorToMatrix( NumericVector v, int nrow, int ncol ){
  v.attr( "dim" ) = Dimension( nrow, ncol );
  return ( v );
}

// [[Rcpp::export]]
List linkClusters( IntegerVector edges, IntegerVector nodes ){
  List g;
  int nedges = edges.size();
  for( int i = 0; i < nedges; i+=2 ){
    if ( i == 0 ){
      g.push_back( IntegerVector::create( edges( i ), edges( i+1 ) ) );
    }
    else{
      int edge1 = -1;
      int edge2 = -1;
      vector< int > remove;
      int len = g.length();
      for ( int j = 0; j < len; j++ ){
        IntegerVector v = g[j];
        bool edge1Found = find( v.begin(), v.end(), edges( i ) ) != v.end();
        bool edge2Found = find( v.begin(), v.end(), edges( i+1 ) ) != v.end();
        if ( edge1Found && edge2Found ){
          edge1 = j;
          edge2 = j;
        }
        else if ( edge1Found && !edge2Found ){
          edge1 = j;
          // check if the second node is in another group
          for ( int k = 0; k < len; k++ ){
            if ( k != j ){
              IntegerVector w = g[k];
              if ( find( w.begin(), w.end(), edges( i+1 ) ) != w.end() ){
                // if found, then push all the nodes in this vector into v
                for ( int l : w ){
                  v.push_back( l );
                }
                g[j] = v;
                // remove this list index from g
                remove.push_back( k );
                edge2 = j;
                break;
              }
            }
          }
        }
        else if ( !edge1Found && edge2Found ){
          edge2 = j;
          // check if the first node is in another group
          for ( int k = 0; k < len; k++ ){
            if ( k != j ){
              IntegerVector w = g[k];
              if ( find( w.begin(), w.end(), edges( i+1 ) ) != w.end() ){
                for ( int l : w ){
                  v.push_back( l );
                }
                g[j] = v;
                remove.push_back( k );
                edge2 = j;
                break;
              }
            }
          }
        }
        if (edge1 > -1 && edge2 > -1)
          break;

      }
      // both nodes are not in g, so add them as a separate group
      if ( edge1 == -1 && edge2 == -1 ){
        g.push_back( IntegerVector::create( edges( i ), edges( i+1 ) ) );
      }
      else if ( edge1 == -1 && edge2 > -1 ){
        IntegerVector e = g[edge2];
        e.push_back( edges( i ) );
        g[edge2] = e;
      }
      else if ( edge1 > -1 && edge2 == -1 ){
        IntegerVector e = g[edge1];
        e.push_back( edges( i+1 ) );
        g[edge1] = e;
      }

      if ( remove.size() > 0 ){
        sort( remove.rbegin(), remove.rend() );
        for ( int r : remove ){
          g.erase( r );
        }
      }
    }
  }

  // Add the nodes that are not linked into the linkedGroup

  int len = nodes.length();
  int glen = g.length();
  for ( int i = 0; i < len; i++ ){
    bool found = false;
    for ( int j = 0; j < glen; j++ ){
      IntegerVector v = g[j];
      if ( find( v.begin(), v.end(), nodes( i ) ) != v.end() ){
        found = true;
        break;
      }
    }
    if ( !found ){
      g.push_back( nodes( i ) );
    }
  }

  return g;
}

// [[Rcpp::export(.encodeNumericLabel)]]
NumericMatrix encodeNumericLabel( NumericVector labels, List code ){
  int l = labels.length();
  int col = as<NumericVector>( code[0] ).length();
  NumericMatrix convert( l, col );
  for ( int i = 0; i < l; i++ ){
    convert( i, _ ) = as<NumericVector>( code[to_string( labels[i] )] );
  }
  return convert;
}

// [[Rcpp::export(.encodeStringLabel)]]
NumericMatrix encodeStringLabel( StringVector labels, List code ){
  int l = labels.length();
  int col = as<NumericVector>( code[0] ).length();
  NumericMatrix convert( l, col );
  for ( int i = 0; i < l; i++ ){
    convert( i, _ ) = as<NumericVector>( code[as<string>( labels[i] )] );
  }
  return convert;
}

void printVector( NumericVector v ){
  int l = v.length();
  for ( int i = 0; i < l; i++ ){
    cout << v[i] << " ";
  }
  cout << endl;
}

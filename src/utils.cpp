/****************************************************************************
 *
 *  utils.cpp
 *  ART utilities
 *
 ****************************************************************************/

#include <Rcpp.h>
using namespace Rcpp;


// sortIndex: Sort the values in descending order and return the indices of the values
NumericVector sortIndex( NumericVector x ){
  NumericVector idx( x.size() );
  std::iota( idx.begin(), idx.end(), 0 );

  std::stable_sort( idx.begin(), idx.end(), [&x]( size_t i1, size_t i2 ) { return x[i1] > x[i2]; } );

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

// appendColumns: append columns to a matrix
NumericMatrix appendColumns( NumericMatrix x, int numCols ){
  int nrow = x.nrow();
  int ncol = x.ncol();
  NumericMatrix v ( nrow, ncol + numCols );
  for ( int i = 0; i < ncol; i++ ){
    v ( _,i ) = x( _,i );
  }
  return v;
}

// appendVector: increase vector length
template <class T> T appendVector( T x, int length ){
  int l = x.length();
  T v ( l + length );
  for ( int i = 0; i < l; i++ ){
    v( i ) = x( i );
  }
  return v;
}
template IntegerVector appendVector( IntegerVector x, int length );
template NumericVector appendVector( NumericVector x, int length );

// joinVectors: Join two vectors together
template <class T> T joinVectors( T v1, T v2 ){
  int l1 = v1.length();
  int l2 = v2.length();
  T v( l1 + l2 );
  for ( int i = 0; i < l1; i++ ){
    v( i ) = v1( i );
  }
  for ( int i = 0; i < l2; i++ ){
    v( i + l1 ) = v2( i );
  }
  return v;
}
template IntegerVector joinVectors( IntegerVector v1, IntegerVector v2 );
template NumericVector joinVectors( NumericVector v1, NumericVector v2 );


NumericMatrix subsetRows( NumericMatrix x, int rows ){
  NumericMatrix y;
  
  if ( rows == 0 ){
    /* return an empty matrix */
    return y;
  }
  
  if ( rows > x.rows() ){
    stop( "Too many rows have been selected." );
  }

  y = x( Range( 0, rows - 1 ), _ );
  return y;
}

NumericMatrix subsetColumns( NumericMatrix x, int cols ){
  NumericMatrix y;
  
  if ( cols == 0 ){
    /* return an empty matrix */
    return y;
  }
  
  if ( cols > x.cols() ){
    stop( "Too many columns have been selected." );
  }
  
  y = x( _, Range( 0, cols - 1 ) );
  return y;
}

IntegerVector subsetVector( IntegerVector x, int length ){
  IntegerVector y;
  if ( length == 0 ){
    return y;
  }
  if ( length > x.length() ){
    stop( "Too many rows have been selected." );
  }
  
  y = x[Range( 0, length - 1 )];
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
      std::vector< int > remove;
      int len = g.length();
      for ( int j = 0; j < len; j++ ){
        IntegerVector v = g[j];
        bool edge1Found = std::find( v.begin(), v.end(), edges( i ) ) != v.end();
        bool edge2Found = std::find( v.begin(), v.end(), edges( i+1 ) ) != v.end();
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
              if ( std::find( w.begin(), w.end(), edges( i+1 ) ) != w.end() ){
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
              if ( std::find( w.begin(), w.end(), edges( i+1 ) ) != w.end() ){
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
      if ( std::find( v.begin(), v.end(), nodes( i ) ) != v.end() ){
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

// [[Rcpp::export(.createDummyCodeMap)]]
List createDummyCodeMap ( StringVector classLabels ){
 
  int l = classLabels.length();
  IntegerVector pos = seq( 0, l );
  List code;
  for ( int i = 0; i < l; i++ ){
    String label = classLabels[i];
    code.push_back( rep( 0, l ), label.get_cstring() );
    as<IntegerVector>( code[label.get_cstring()] )( pos[i] ) = 1;
  }

  return code;
}

// [[Rcpp::export(.encodeNumericLabel)]]
NumericMatrix encodeNumericLabel( IntegerVector labels, List code ){
  int l = labels.length();
  int col = as<IntegerVector>( code[0] ).length();
  NumericMatrix convert( l, col );
  for ( int i = 0; i < l; i++ ){
    String s = std::to_string( labels[i] );
    convert( i, _ ) = as<IntegerVector>( code[s] );
  }
  return convert;
}

// [[Rcpp::export(.encodeStringLabel)]]
NumericMatrix encodeStringLabel( StringVector labels, List code ){
  int l = labels.length();
  int col = as<IntegerVector>( code[0] ).length();
  NumericMatrix convert( l, col );
  for ( int i = 0; i < l; i++ ){
    String s = labels[i];
    convert( i, _ ) = as<IntegerVector>( code[s] );
  }
  return convert;
}

// [[Rcpp::export(.decode)]]
StringVector decode (NumericMatrix dummyClasses, List dummyCode){
  int rows = dummyClasses.rows();
  int l = dummyCode.length();
  StringVector labels( rows );
  for ( int i = 0; i < rows; i++ ){
    for ( int j = 0; j < l; j++ ){
      String s = as<CharacterVector>( dummyCode.names() )[j];
      int d = sum( abs( as<NumericVector>( dummyCode[s.get_cstring()] ) - dummyClasses( i, _ ) ) );
      if ( d == 0 ){
        labels[i] = s.get_cstring();
        break;
      } 
    }
  }
  return labels;
}

// Check if two vectors are the same.
bool equal( NumericVector x, NumericVector y ){
  if ( x.length() != y.length() ){
    stop( "The two vectors are not the same length." );
  }
  int l = x.length();
  bool equal = true;
  for ( int i = 0; i < l; i++ ){
    if ( x[i] != y[i] ) {
      equal = false;
      break;
    }
  }
  return equal;
}

void printVector( NumericVector v ){
  int l = v.length();
  for ( int i = 0; i < l; i++ ){
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

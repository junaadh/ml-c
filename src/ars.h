#ifndef ARS_H_
#define ARS_H_

// include statements
#include <math.h>
#include <stddef.h>
#include <stdio.h>

// if ARS_MALLOC not defined use std malloc
#ifndef ARS_MALLOC
#include <stdlib.h>
#define ARS_MALLOC malloc
#endif // ARS_MALLOC

// if ARS_ASSERT not defined use std assert
#ifndef ARS_ASSERT
#include <assert.h>
#define ARS_ASSERT assert
#endif // ARS_ASSERT

// lightweight structure of matrix
typedef struct {
  // number of rows in matrix
  size_t rows;
  // number of cols in matrix
  size_t cols;
  // pointer to the start of matrix
  float *es;
} Mat;

//
// function declarations
//

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m, const char *name);
void mat_fill(Mat m, float x);
void mat_sig(Mat m);

// matrix value at MAT_AT(source, rows, cols)
#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]
// takes in a matrix and prints formated print statement
#define MAT_PRINT(m) mat_print(m, #m)

#endif // ARS_H_

// following logic only avaliable if ARS_IMPLEMENTATION defined
#ifdef ARS_IMPLEMENTATION

// returns a randomized float from range 0-1
float rand_float(void) { return (float)rand() / RAND_MAX; }

// activation function which limits the value from 0-1
float sigmoidf(float x) { return 1.f / (1.f + expf(x)); }

// allocate space for the matrix in heap using malloc if ARS_MALLOC is not
// defined if ARS_MALLOC defined use the custom malloc
Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.es = ARS_MALLOC(sizeof(*m.es) * rows * cols);
  ARS_ASSERT(m.es != NULL);
  return m;
}

// populates matrix with random values
// m is the matrix to populate and low and high refer to the range
void mat_rand(Mat m, float low, float high) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low;
    }
  }
}

// multiplies matrices a and b storing the resulting matrix in dest
// a and b need to have same a cols and b rows need to be same
// and dest size is a rows and b cols
void mat_dot(Mat dest, Mat a, Mat b) {
  ARS_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  ARS_ASSERT(dest.rows == a.rows);
  ARS_ASSERT(dest.cols == b.cols);
  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      MAT_AT(dest, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

// adds two matrices, a is added to dest
// dest and a needs to be equal size
void mat_sum(Mat dest, Mat a) {
  ARS_ASSERT(dest.rows == a.rows);
  ARS_ASSERT(dest.cols == a.cols);
  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      MAT_AT(dest, i, j) += MAT_AT(a, i, j);
    }
  }
}

// prints the matrix
void mat_print(Mat m, const char *name) {
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("    %f", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

// fills matrix with float x
// used for debugging and testing
void mat_fill(Mat m, float x) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = x;
    }
  }
}

// function for matrix activation
// performs sigmoidf() on each value in the matrix
void mat_sig(Mat m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}

#endif // ARS_IMPLEMENTATION

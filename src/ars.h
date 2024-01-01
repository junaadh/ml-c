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

//
// macros
//
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
// matrix value at MAT_AT(source, rows, cols)
#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
// takes in a matrix and prints formated print statement
#define MAT_PRINT(m) mat_print(m, #m, 0)
// takes in a network and prints foramted print statement
#define NET_PRINT(net) net_print(net, #net)

// lightweight structure of matrix
typedef struct {
  // number of rows in matrix
  size_t rows;
  // number of cols in matrix
  size_t cols;
  // the strides to reach next row
  size_t stride;
  // pointer to the start of matrix
  float *es;
} Mat;

//
// matrix function declarations
//

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dest, Mat src);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m, const char *name, size_t padding);
void mat_fill(Mat m, float x);
void mat_sig(Mat m);

// structure of the neural network
typedef struct {
  size_t count;
  Mat *ws;
  Mat *bs;
  Mat *as; // the amount of activations is count++
} Net;

#define NET_INPUT(net) (net).as[0]
#define NET_OUTPUT(net) (net).as[(net).count]

//
// neural network function declarations
//
Net net_alloc(size_t *arch, size_t arch_count);
void net_print(Net net, const char *name);
void net_rand(Net net, float low, float high);
void net_forward(Net net);
float net_cost(Net net, Mat ti, Mat to);
void net_diff(Net net, Net grad, float eps, Mat ti, Mat to);
void net_learn(Net net, Net grad, float rate);

#endif // ARS_H_

// following logic only avaliable if ARS_IMPLEMENTATION defined
#ifdef ARS_IMPLEMENTATION

//
// matrix
//
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
  m.stride = cols;
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

Mat mat_row(Mat m, size_t row) {
  return (Mat){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
      .es = &MAT_AT(m, row, 0),
  };
}

void mat_copy(Mat dest, Mat src) {
  ARS_ASSERT(dest.rows == src.rows);
  ARS_ASSERT(dest.cols == src.cols);
  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      MAT_AT(dest, i, j) = MAT_AT(src, i, j);
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
void mat_print(Mat m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int)padding, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("%*s  ", (int)padding, "");
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f  ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
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

//
// Neural Netwoik
//
//
Net net_alloc(size_t *arch, size_t arch_count) {
  ARS_ASSERT(arch_count > 0);

  Net net;
  net.count = arch_count - 1;

  net.ws = ARS_MALLOC(sizeof(*net.ws) * net.count);
  ARS_ASSERT(net.ws != NULL);
  net.bs = ARS_MALLOC(sizeof(*net.bs) * net.count);
  ARS_ASSERT(net.bs != NULL);
  net.as = ARS_MALLOC(sizeof(*net.as) * (net.count + 1));
  ARS_ASSERT(net.as != NULL);

  net.as[0] = mat_alloc(1, arch[0]);
  for (size_t i = 1; i < arch_count; ++i) {
    net.ws[i - 1] = mat_alloc(net.as[i - 1].cols, arch[i]);
    net.bs[i - 1] = mat_alloc(1, arch[i]);
    net.as[i] = mat_alloc(1, arch[i]);
  }
  return net;
}

void net_print(Net net, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < net.count; ++i) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(net.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    mat_print(net.bs[i], buf, 4);
  }
  printf("]\n");
}

void net_rand(Net net, float low, float high) {
  for (size_t i = 0; i < net.count; ++i) {
    mat_rand(net.ws[i], low, high);
    mat_rand(net.bs[i], low, high);
  }
}

void net_forward(Net net) {
  for (size_t i = 0; i < net.count; ++i) {
    mat_dot(net.as[i + 1], net.as[i], net.ws[i]);
    mat_sum(net.as[i + 1], net.bs[i]);
    mat_sig(net.as[i + 1]);
  }
}

float net_cost(Net net, Mat ti, Mat to) {
  ARS_ASSERT(ti.rows == to.rows);
  ARS_ASSERT(to.cols == NET_OUTPUT(net).cols);
  size_t n = ti.rows;

  float c = 0;
  for (size_t i = 0; i < n; ++i) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);

    mat_copy(NET_INPUT(net), x);
    net_forward(net);
    size_t q = to.cols;
    for (size_t j = 0; j < q; ++j) {
      float d = MAT_AT(NET_OUTPUT(net), 0, j) - MAT_AT(y, 0, j);
      c += d * d;
    }
  }
  return c / n;
}

void net_diff(Net net, Net grad, float eps, Mat ti, Mat to) {
  float saved;
  float c = net_cost(net, ti, to);
  for (size_t i = 0; i < net.count; ++i) {
    for (size_t j = 0; j < net.ws[i].rows; ++j) {
      for (size_t k = 0; k < net.ws[i].cols; ++k) {
        saved = MAT_AT(net.ws[i], j, k);
        MAT_AT(net.ws[i], j, k) += eps;
        MAT_AT(grad.ws[i], j, k) = (net_cost(net, ti, to) - c) / eps;
        MAT_AT(net.ws[i], j, k) = saved;
      }
    }
    for (size_t j = 0; j < net.bs[i].rows; ++j) {
      for (size_t k = 0; k < net.bs[i].cols; ++k) {
        saved = MAT_AT(net.bs[i], j, k);
        MAT_AT(net.bs[i], j, k) += eps;
        MAT_AT(grad.bs[i], j, k) = (net_cost(net, ti, to) - c) / eps;
        MAT_AT(net.bs[i], j, k) = saved;
      }
    }
  }
}

void net_learn(Net net, Net grad, float rate) {
  for (size_t i = 0; i < net.count; ++i) {
    for (size_t j = 0; j < net.ws[i].rows; ++j) {
      for (size_t k = 0; k < net.ws[i].cols; ++k) {
        MAT_AT(net.ws[i], j, k) -= rate * MAT_AT(grad.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < net.bs[i].rows; ++j) {
      for (size_t k = 0; k < net.bs[i].cols; ++k) {
        MAT_AT(net.bs[i], j, k) -= rate * MAT_AT(grad.bs[i], j, k);
      }
    }
  }
}
#endif // ARS_IMPLEMENTATION

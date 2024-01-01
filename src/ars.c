#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// sepcify that we want the ARS_INPLEMENTATION from ars.h
#define ARS_IMPLEMENTATION
#include "ars.h"

typedef struct {
  size_t rows;
  size_t cols;
  float data[];
} train_set_t;

// clang-format off
// training data set
// formatting turned off for legibility coz clangd does single line arrays
train_set_t td_or = {
  .rows = 4, 
  .cols = 3, 
  .data = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
  }
};
train_set_t td_and = {
  .rows = 4, 
  .cols = 3, 
  .data = {
  0, 0, 0,
  0, 1, 0,
  1, 0, 0,
  1, 1, 1,
  }
};
train_set_t td_nand = {
  .rows = 4, 
  .cols = 3, 
  .data = {
  0, 0, 1,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
  }
};
train_set_t td_xor = {
  .rows = 4, 
  .cols = 3, 
  .data = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
  }
};
train_set_t td_nor = {
  .rows = 4, 
  .cols = 3, 
  .data = {
  0, 0, 1,
  0, 1, 0,
  1, 0, 0,
  1, 1, 0,
  }
};
// clang-format on

int main(void) {
  // set the randomizing token
  srand(time(0));

  train_set_t *td = &td_nand;

  // set epselon and rate for learning
  float eps = 1e-1;
  float rate = 1e-1;

  // the stride at which you want to form a  submatrix
  size_t stride = 3;
  // the number of rows in the training data set

  // sub matrix of only training data inputs
  Mat ti = {.rows = td->rows, .cols = 2, .stride = stride, .es = td->data};
  // sub matrix of training data output
  Mat to = {.rows = td->rows, .cols = 1, .stride = stride, .es = td->data + 2};

  // architecture of the neural network
  // arch[0] represents input params according to training data
  // arch[ARRAY_LEN(arch) - 1] represents the output according to the training
  // data
  size_t arch[] = {
      2,
      2,
      10,
      1,
  };

  // base neural network
  Net net = net_alloc(arch, ARRAY_LEN(arch));
  // gradient neural network
  Net grad = net_alloc(arch, ARRAY_LEN(arch));

  // initialize random values in the network
  net_rand(net, 0, 1);

  // printing the cost as the network learns
  printf("\ninitial cost = %f\n", net_cost(net, ti, to));
  for (size_t i = 0; i < 100 * 1000; ++i) {
    net_diff(net, grad, eps, ti, to);
    net_learn(net, grad, rate);
// make if 0 if want to disable tracing
#if 1
    printf("%zu: cost = %f\n", i, net_cost(net, ti, to));
#endif
  }

// make 0 if dont wanna see the test case
#if 1
  printf("\n------------------------------------------\n");

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      // set the inputs
      MAT_AT(NET_INPUT(net), 0, 0) = i;
      MAT_AT(NET_INPUT(net), 0, 1) = j;
      // use the model to get the output
      net_forward(net);
      // print the output
      printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NET_OUTPUT(net), 0, 0));
    }
  }
#endif

  return 0;
}

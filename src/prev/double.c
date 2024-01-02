#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {{0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}};

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

// differential old one
float cost(float w) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0];
    float y = x * w;
    float d = y - train[i][1];
    result += d * d;
  }
  result /= train_count;
  return result;
}

// derivative new one
float dcost(float w) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0];
    float y = train[i][1];
    result += 2 * (x * w - y) * x;
  }
  result /= train_count;
  return result;
}

int main() {
  // srand(time(0));
  srand(69);
  float w = rand_float() * 10.0f;
  float rate = 1e-1;
  printf("og cost: %f;  og w = %f\n", dcost(w), w);

  for (size_t i = 0; i < 50; ++i) {
#if 0     
    float eps = 1e-1;
    float c = cost(w);
    float dw = (cost(w + eps) - c) / eps;
#else
    float dw = dcost(w);
#endif
    w -= rate * dw;
    printf("%zu: cost: %f; w = %f\n", i, cost(w), w);
  }

  printf("--------------------------\n");

  //
  // according to the used train dataset
  // w needs to be 2
  //

  printf("w = %f\n", w);

  for (size_t i = 1; i <= 10; i += 2) {
    printf("%zu > %f\n", i, i * w);
  }

  return 0;
}

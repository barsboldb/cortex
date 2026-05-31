#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>

#define tensor_at(t, i, j) ((t)->data[(i) * (t)->cols + (j)])

typedef struct tensor_t {
  float *data;
  int    rows;
  int    cols;
} Tensor;

Tensor *tensor_create(int rows, int cols); 
Tensor *tensor_copy(const Tensor *t);
void tensor_init_xa(Tensor *t, int fan_in, int fan_out);
void tensor_init_he(Tensor *t, int fan_in);
void tensor_free(Tensor *t);

void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_matmul_bt(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_matmul_at(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_add(Tensor *a, Tensor *b);
void tensor_scale(Tensor *a, float s);
void tensor_sum_rows(Tensor *a, Tensor *out);
void tensor_add_bias(Tensor *a, const Tensor *bias);

void tensor_print(const Tensor *t, const char *label);

#endif

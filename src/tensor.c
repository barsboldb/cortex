#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "tensor.h"

static float rand_uniform(float limit) {
  return ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
}

Tensor *tensor_create(int rows, int cols) {
  Tensor *t = malloc(sizeof(Tensor));
  t->data = calloc(rows * cols, sizeof(float));
  t->rows = rows;
  t->cols = cols;
  return t;
}

Tensor *tensor_copy(const Tensor *t) {
  int size = t->rows * t->cols * sizeof(float);
  Tensor *r = tensor_create(t->rows, t->cols);
  memcpy(r->data, t->data, size);
  return r;
}

void tensor_free(Tensor *t) {
  if (!t) return;

  free(t->data);
  free(t);
}

void tensor_init_xa(Tensor *t, int fan_in, int fan_out) {
  float limit = sqrtf(6.0f / (fan_in + fan_out));
  for (int i = 0; i < t->rows; ++i)
    for (int j = 0; j < t->cols; ++j)
      tensor_at(t, i, j) = rand_uniform(limit);
}

void tensor_init_he(Tensor *t, int fan_in) {
  float limit = sqrtf(6.0f / fan_in);
  for (int i = 0; i < t->rows; ++i)
    for (int j = 0; j < t->cols; ++j)
      tensor_at(t, i, j) = rand_uniform(limit);
}

void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out) {
  assert(a->cols == b->rows);
  assert(out->rows == a->rows);
  assert(out->cols == b->cols);

  for (int i = 0; i < a->rows; ++i) {
    for (int j = 0; j < b->cols; ++j) {
      float sum = 0;
      for (int k = 0; k < a->cols; ++k) {
        sum += tensor_at(a, i, k) * tensor_at(b, k, j);
      }
      tensor_at(out, i, j) = sum;
    }
  }
}

void tensor_matmul_bt(const Tensor *a, const Tensor *b, Tensor *out) {
  assert(a->cols == b->cols);
  assert(out->rows == a->rows);
  assert(out->cols == b->rows);

  for (int i = 0; i < a->rows; ++i) {
    for (int j = 0; j < b->rows; ++j) {
      float sum = 0;
      for (int k = 0; k < a->cols; ++k) {
        sum += tensor_at(a, i, k) * tensor_at(b, j, k);
      }
      tensor_at(out, i, j) = sum;
    }
  }
}

void tensor_matmul_at(const Tensor *a, const Tensor *b, Tensor *out) {
  assert(a->rows == b->rows);
  assert(out->rows == a->cols);
  assert(out->cols == b->cols);

  for (int i = 0; i < a->cols; i++) {
    for (int j = 0; j < b->cols; j++) {
      float sum = 0;
      for (int k = 0; k < b->rows; k++)
        sum += (tensor_at(a, k, i) * tensor_at(b, k, j));
      tensor_at(out, i, j) = sum;
    }
  }
}

void tensor_add_bias(Tensor *a, const Tensor *bias) {
  assert(bias->rows == 1);
  assert(bias->cols == a->cols);

  for (int i = 0; i < a->rows; i++)
    for (int j = 0; j < a->cols; j++)
      tensor_at(a, i, j) += tensor_at(bias, 0, j);
}

void tensor_add(Tensor *a, Tensor *b) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      tensor_at(a, i, j) += tensor_at(b, i, j);
    }
  }
}

void tensor_scaled_add(Tensor *a, Tensor *b, float s) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      tensor_at(a, i, j) += tensor_at(b, i, j) * s;
    }
  }
}

void tensor_scale(Tensor *a, float s) {
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      a->data[i * a->cols + j] *= s;
    }
  }
}

void tensor_sum_rows(Tensor *a, Tensor *out) {
  for (int i = 0; i < a->cols; i++) {
    float sum = 0;
    for (int j = 0; j < a->rows; j++) {
      sum += tensor_at(a, j, i);
    }
    out->data[i] = sum;
  }
}

void tensor_hadamard(Tensor *a, Tensor *b) {
  assert(a->rows == b->rows);
  assert(a->cols == b->cols);

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      tensor_at(a, i, j) *= tensor_at(b, i, j);
    }
  }
}

void tensor_print(const Tensor *t, const char *label) {
  printf("%s [%d x %d]\n", label, t->rows, t->cols);
  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      printf("%3f ", tensor_at(t, i, j));
    }
    printf("\n");
  }
}


void tensor_zero(Tensor *t) {
  if (!t) return;

  memset(t->data, 0, t->rows * t->cols * sizeof(float));
}


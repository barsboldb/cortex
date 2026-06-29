#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

typedef enum {
  OPT_SGD,
  OPT_ADAM,
} OptKind;

typedef struct {
  OptKind kind;
  int     n_params;
  float   lr;
  Tensor **p;
  union {
    struct {
      float    m;
      Tensor **v;
    } sgd;
    struct {
      float beta1, beta2, epsilon;
      int   step;
      Tensor **m;
      Tensor **v;
    } adam;
  };
} Optimizer;

Optimizer *opt_sdg_create(float lr, float momentum, int n_params, Tensor **params);
Optimizer *opt_adam_create(float lr, float b1, float b2, float epsilon, int n_params, Tensor **params);
void       opt_step(Optimizer *o, Tensor **grad);

void opt_free(Optimizer *o);

#endif

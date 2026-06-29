#include <math.h>
#include "optimizer.h"
#include "tensor.h"

static inline void sgd_step(Optimizer *o, Tensor **grad) {
  if (!o->sgd.v) {
    o->sgd.v = malloc(sizeof(Tensor *) * o->n_params) ;   
    for (int i = 0; i < o->n_params; i++) {
      o->sgd.v[i] = tensor_create(o->p[i]->rows, o->p[i]->cols);
      tensor_zero(o->sgd.v[i]);
    }
  }
  for (int i = 0; i < o->n_params; i++) {
    tensor_scale(o->sgd.v[i], o->sgd.m);
    tensor_add(o->sgd.v[i], grad[i]);
    tensor_scaled_add(o->p[i], o->sgd.v[i], -o->lr);
  }
} 

static inline void adam_step(Optimizer *o, Tensor **grad) {
  o->adam.step++;
  if (!o->adam.m) {
    o->adam.m = malloc(sizeof(Tensor *) * o->n_params);
    o->adam.v = malloc(sizeof(Tensor *) * o->n_params);
    for (int i = 0; i < o->n_params; i++) {
      int rows = o->p[i]->rows;
      int cols = o->p[i]->cols;
      o->adam.m[i] = tensor_create(rows, cols);
      tensor_zero(o->adam.m[i]);

      o->adam.v[i] = tensor_create(rows, cols);
      tensor_zero(o->adam.v[i]);
    }
  }

  for (int i = 0; i < o->n_params; i++) {
    tensor_scale(o->adam.m[i], o->adam.beta1);
    tensor_scaled_add(o->adam.m[i], grad[i], 1.0f - o->adam.beta1);

    Tensor *grad_sq = tensor_copy(grad[i]);
    tensor_hadamard(grad_sq, grad[i]);
    tensor_scale(o->adam.v[i], o->adam.beta2);
    tensor_scaled_add(o->adam.v[i], grad_sq, 1.0f - o->adam.beta2);
    tensor_free(grad_sq);

    float m_scale = 1.0f / (1.0f - powf(o->adam.beta1, o->adam.step));
    float v_scale = 1.0f / (1.0f - powf(o->adam.beta2, o->adam.step));

    int n = o->p[i]->rows * o->p[i]->cols;
    for (int j = 0; j < n; j++) {
      float m_hat = o->adam.m[i]->data[j] * m_scale;
      float v_hat = o->adam.v[i]->data[j] * v_scale;
      o->p[i]->data[j] -= o->lr * m_hat / (sqrtf(v_hat) + o->adam.epsilon);
    }
  }
}

Optimizer *opt_sdg_create(float lr, float momentum, int n_params, Tensor **params) {
  Optimizer *o = malloc(sizeof(Optimizer));

  o->kind     = OPT_SGD;
  o->lr       = lr;
  o->n_params = n_params;
  o->sgd.m    = momentum;
  o->sgd.v    = NULL;
  o->p        = params;

  return o;
}

Optimizer *opt_adam_create(float lr, float b1, float b2, float epsilon, int n_params, Tensor **params) {
  Optimizer *o = malloc(sizeof(Optimizer));

  o->kind = OPT_ADAM;
  o->lr           = lr;
  o->n_params     = n_params;
  o->p            = params;
  o->adam.beta1   = b1;
  o->adam.beta2   = b2;
  o->adam.epsilon = epsilon;
  o->adam.step    = 0;
  o->adam.m       = NULL;
  o->adam.v       = NULL;

  return o;
}

void opt_step(Optimizer *o, Tensor **grad) {
  switch (o->kind) {
  case OPT_SGD: return sgd_step(o, grad); 
  case OPT_ADAM: return adam_step(o, grad);
  default: return;
  }
}

void opt_free(Optimizer *o) {
  for (int i = 0; i < o->n_params; i++) {
    if (o->kind == OPT_SGD && o->sgd.v) {
      tensor_free(o->sgd.v[i]);
      continue;
    }
    if (o->adam.m) {
      tensor_free(o->adam.m[i]);
    }
    if (o->adam.v) {
      tensor_free(o->adam.v[i]);
    }
  }
  if (o->kind == OPT_SGD) {
    free(o->sgd.v);
  } else {
    free(o->adam.m);
    free(o->adam.v);
  }
  free(o);
}

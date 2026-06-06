#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "tensor.h"

typedef enum {
  ACT_RELU,
  ACT_LRELU,
  ACT_SOFTMAX,
  ACT_SIGMOID,
  ACT_TANH,
  ACT_LINEAR,
} ActKind;

void activation_forward(ActKind kind, Tensor *t);
void activation_backward(ActKind kind, const Tensor *out, Tensor *grad);

#endif

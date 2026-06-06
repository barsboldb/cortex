#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef enum {
  LOSS_MSE,
  LOSS_CE,
} LossKind;

float loss_forward(LossKind kind, const Tensor *pred, const Tensor *target, Tensor *grad);

#endif

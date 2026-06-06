#include "loss.h"
#include <math.h>

#define eps (1e-7f)

static inline float loss_mse(const Tensor *pred, const Tensor *target, Tensor *grad) {
  int n = pred->rows * pred->cols;
  float loss = 0;
  for (int i = 0; i < n; i++) {
    float dl = pred->data[i] - target->data[i];
    loss += dl * dl;
    grad->data[i] = 2 * (dl) / n;
  }

  return loss / n;
}

static inline float loss_ce(const Tensor *pred, const Tensor *target, Tensor *grad) {
  int n = pred->rows * pred->cols;
  float loss = 0;
  for (int i = 0; i < n; i++) {
    loss += -target->data[i] * logf(pred->data[i] + eps);
    grad->data[i] = (pred->data[i] - target->data[i]) / pred->rows;
  }

  return loss / pred->rows;
}

float loss_forward(LossKind kind, const Tensor *pred, const Tensor *target, Tensor *grad) {
  switch (kind) {
    case LOSS_MSE:
      return loss_mse(pred, target, grad);
    case LOSS_CE:
      return loss_ce(pred, target, grad);
    default:
      return 0.0f;
  }
}

#include <assert.h>
#include <math.h>
#include "activation.h"

#define max(x, y) ((x) < (y) ? (y) : (x))

#define relu(x) (max(0, (x)))
#define sigmoid(x) (1 / (1 + expf(-(x))))
#define lrelu(x) (max(0.01f * x, (x)))

#define drelu(x) ((x) > 0 ? 1 : 0)
#define dsigmoid(x) ((x) * (1 - (x)))
#define dtanh(x) (1 - (x) * (x))
#define dlrelu(x) ((x) > 0 ? 1 : 0.01)

static void act_relu(Tensor *t) {
  int n = t->rows * t->cols;
  for (int i = 0; i < n; i++) {
    t->data[i] = relu(t->data[i]);
  }
}

static void act_sigmoid(Tensor *t) {
  int n = t->rows * t->cols;
  for (int i = 0; i < n; i++) {
    t->data[i] = sigmoid(t->data[i]);
  }
}

static void act_tanh(Tensor *t) {
  int n = t->rows * t->cols;
  for (int i = 0; i < n; i++) {
    t->data[i] = tanhf(t->data[i]);
  }
}

static void act_lrelu(Tensor *t) {
  int n = t->rows * t->cols;
  for (int i = 0; i < n; i++) {
    t->data[i] = lrelu(t->data[i]);
  }
}

static void act_softmax(Tensor *t) {
  for (int i = 0; i < t->rows; i++) {
    float row_max = tensor_at(t, i, 0);
    float denom = 0; 

    for (int j = 0; j < t->cols; j++)
      if (tensor_at(t, i, j) > row_max)
        row_max = tensor_at(t, i, j);

    for (int j = 0; j < t->cols; j++)
      denom += expf(tensor_at(t, i, j) - row_max);

    for (int j = 0; j < t->cols; j++)
      tensor_at(t, i, j) = expf(tensor_at(t, i, j) - row_max) / denom;
  }
}

static void der_relu(const Tensor *t, Tensor *grad) {
  int n = grad->rows * grad->cols;
  for (int i = 0; i < n; i++) {
    grad->data[i] = grad->data[i] * drelu(t->data[i]);
  }
}

static void der_sigmoid(const Tensor *t, Tensor *grad) {
  int n = grad->rows * grad->cols;
  for (int i = 0; i < n; i++) {
    grad->data[i] = grad->data[i] * dsigmoid(t->data[i]);
  }
}

static void der_tanh(const Tensor *t, Tensor *grad) {
  int n = grad->rows * grad->cols;
  for (int i = 0; i < n; i++) {
    grad->data[i] = grad->data[i] * dtanh(t->data[i]);
  }
}

static void der_lrelu(const Tensor *t, Tensor *grad) {
  int n = grad->rows * grad->cols;
  for (int i = 0; i < n; i++) {
    grad->data[i] = grad->data[i] * dlrelu(t->data[i]);
  }
}

void activation_forward(ActKind kind, Tensor *t) {
  switch (kind) {
    case ACT_RELU:
      return act_relu(t);
    case ACT_SIGMOID:
      return act_sigmoid(t);
    case ACT_TANH:
      return act_tanh(t);
    case ACT_LRELU:
      return act_lrelu(t);
    case ACT_SOFTMAX:
      return act_softmax(t);
    default: return;
  }

  assert(0);
}

void activation_backward(ActKind kind, const Tensor *out, Tensor *grad) {
  switch (kind) {
    case ACT_RELU:
      return der_relu(out, grad);
    case ACT_SIGMOID:
      return der_sigmoid(out, grad);
    case ACT_TANH:
      return der_tanh(out, grad);
    case ACT_LRELU:
      return der_lrelu(out, grad);
    default: return;
  }
}


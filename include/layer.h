#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "activation.h"

typedef struct {
  Tensor *weights;
  Tensor *bias;
  ActKind kind;
  Tensor *grad_weights; 
  Tensor *grad_bias;
  Tensor *in_cache;
  Tensor *out_cache;

  int in_feature;
  int out_feature;
} Layer;

Layer *layer_create(int in, int out, ActKind kind);
void   layer_free(Layer *layer);

Tensor *layer_forward(Layer *l, const Tensor *input);
Tensor *layer_backward(Layer *l, const Tensor *grad);
void    layer_zero_grad(Layer *l);

#endif

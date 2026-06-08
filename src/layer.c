#include <stdlib.h>
#include "layer.h"

Layer *layer_create(int in, int out, ActKind kind) {
  Layer *l = malloc(sizeof(Layer));

  l->in_feature   = in;
  l->out_feature  = out;
  l->kind         = kind;
  l->weights      = tensor_create(out, in);
  l->bias         = tensor_create(1, out);
  l->grad_weights = tensor_create(out, in);
  l->grad_bias    = tensor_create(1, out);
  l->in_cache     = NULL;
  l->out_cache    = NULL;

  if (kind == ACT_RELU || kind == ACT_LRELU)
    tensor_init_he(l->weights, in);
  else
    tensor_init_xa(l->weights, in, out);

  return l;
}

void layer_free(Layer *l) {
  tensor_free(l->weights);
  tensor_free(l->bias);
  tensor_free(l->grad_weights);
  tensor_free(l->grad_bias);
  tensor_free(l->in_cache);
  tensor_free(l->out_cache);
}

Tensor *layer_forward(Layer *l, const Tensor *input) {
  Tensor *out = tensor_create(input->rows, l->out_feature); 
  tensor_matmul_bt(input, l->weights, out);
  tensor_add_bias(out, l->bias);
  activation_forward(l->kind, out);

  if (l->in_cache)  tensor_free(l->in_cache);
  if (l->out_cache) tensor_free(l->out_cache);

  l->in_cache  = tensor_copy(input);
  l->out_cache = tensor_copy(out);
  return out;
}

Tensor *layer_backward(Layer *l, const Tensor *grad) {
  Tensor *g = tensor_copy(grad);
  activation_backward(l->kind, l->out_cache, g);

  Tensor *tmp = tensor_create(l->out_feature, l->in_feature);

  tensor_matmul_at(g, l->in_cache, tmp);
  tensor_add(l->grad_weights, tmp);

  Tensor *tmp_bias = tensor_create(1, l->out_feature);
  tensor_sum_rows(g, tmp_bias);
  tensor_add(l->grad_bias, tmp_bias);

  tensor_free(tmp_bias);
  tensor_free(tmp);

  Tensor *grad_input = tensor_create(grad->rows, l->in_feature);
  tensor_matmul(g, l->weights, grad_input);

  tensor_free(g);
  return grad_input;
}

void layer_zero_grad(Layer *l) {
  tensor_zero(l->grad_weights);
  tensor_zero(l->grad_bias);
}

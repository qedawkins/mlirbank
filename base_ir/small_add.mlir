!tensor_t = tensor<128xf32>

func.func @forward(
    %A : !tensor_t, %B : !tensor_t) -> !tensor_t {
  %C = tensor.empty() : !tensor_t
  %0 = linalg.add ins(%A, %B : !tensor_t, !tensor_t)
                     outs(%C : !tensor_t) -> !tensor_t
  return %0 : !tensor_t
}

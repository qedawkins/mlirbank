!tensor_t = tensor<1024x1024xf32>

func.func @forward() -> !tensor_t {
  %A = util.unfoldable_constant dense<1.0> : !tensor_t
  %B = util.unfoldable_constant dense<2.0> : !tensor_t
  %C = tensor.empty() : !tensor_t
  %0 = linalg.add ins(%A, %B : !tensor_t, !tensor_t)
                     outs(%C : !tensor_t) -> !tensor_t
  return %0 : !tensor_t
}

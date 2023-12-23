!tensor_t = tensor<32x2xf16>

func.func @forward(%A : !tensor_t) -> !tensor_t {
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %A, %c1 : !tensor_t
  //%empty = tensor.empty(%dim) : !tensor_t
  %empty = tensor.empty() : !tensor_t
  %S = linalg.softmax dimension(1) ins(%A : !tensor_t) outs(%empty : !tensor_t) -> !tensor_t
  return %S : !tensor_t
}



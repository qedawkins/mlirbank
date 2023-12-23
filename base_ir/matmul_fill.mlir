!A_size = tensor<256x256xf16>
!B_size = tensor<256x256xf16>
!C_size = tensor<256x256xf16>

func.func @matmul_static(
    %A : !A_size, %B : !B_size) -> !C_size {
  %cst = arith.constant 0.000000e+00 : f16
  %empty = tensor.empty() : !C_size
  %C = linalg.fill ins(%cst : f16) outs(%empty : !C_size) -> !C_size
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

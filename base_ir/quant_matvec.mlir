func.func @forward(%arg0: tensor<4096x32xf16>, %arg1: tensor<4096x32xf16>, %arg2: tensor<32x128xf16>) -> tensor<4096xf16> {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %unfoldable = util.unfoldable_constant dense<1> : tensor<4096x32x128xi4>
  %24 = tensor.empty() : tensor<4096xf16>
  %25 = tensor.empty() : tensor<4096x32x128xf16>
  %26 = linalg.fill ins(%cst : f16) outs(%24 : tensor<4096xf16>) -> tensor<4096xf16>
  %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%unfoldable, %arg0, %arg1 : tensor<4096x32x128xi4>, tensor<4096x32xf16>, tensor<4096x32xf16>) outs(%25 : tensor<4096x32x128xf16>) {
  ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
    %29 = arith.extui %in : i4 to i32
    %30 = arith.uitofp %29 : i32 to f16
    %31 = arith.subf %30, %in_1 : f16
    %32 = arith.mulf %31, %in_0 : f16
    linalg.yield %32 : f16
  } -> tensor<4096x32x128xf16>
  %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg2, %27 : tensor<32x128xf16>, tensor<4096x32x128xf16>) outs(%26 : tensor<4096xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %29 = arith.mulf %in, %in_0 : f16
    %30 = arith.addf %29, %out : f16
    linalg.yield %30 : f16
  } -> tensor<4096xf16>
  return %28 : tensor<4096xf16>
}

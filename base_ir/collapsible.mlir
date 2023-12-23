func.func @forward_dispatch_342_generic_2x32x40x64_f32(%4: tensor<2x32x40x64xf32>) -> tensor<2x32xf32> {
  %cst = arith.constant 2.560000e+03 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<2x32xf32>
  %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<2x32xf32>) -> tensor<2x32xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%4 : tensor<2x32x40x64xf32>) outs(%6 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.addf %in, %out : f32
    linalg.yield %9 : f32
  } -> tensor<2x32xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<2x32xf32>) outs(%5 : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %9 = arith.divf %in, %cst : f32
    linalg.yield %9 : f32
  } -> tensor<2x32xf32>
  return %8 : tensor<2x32xf32>
}

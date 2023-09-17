func.func @forward(%arg0: tensor<4096xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %7 = tensor.empty() : tensor<f32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%arg0 : tensor<4096xf32>) outs(%8 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %10 = math.powf %in, %cst_0 : f32
    %11 = arith.addf %10, %out : f32
    linalg.yield %11 : f32
  } -> tensor<f32>
  return %9 : tensor<f32>
}

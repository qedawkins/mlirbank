#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @forward(%arg0: tensor<1x32000xf16>) -> (tensor<1xf16>, tensor<1xi64>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFC00 : f16
    %0 = tensor.empty() : tensor<1xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
    %2 = tensor.empty() : tensor<1xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<1xf16>) -> tensor<1xf16>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x32000xf16>) outs(%3, %1 : tensor<1xf16>, tensor<1xi64>) {
    ^bb0(%in: f16, %out: f16, %out_0: i64):
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.maximumf %in, %out : f16
      %8 = arith.cmpf ogt, %in, %out : f16
      %9 = arith.select %8, %6, %out_0 : i64
      linalg.yield %7, %9 : f16, i64
    } -> (tensor<1xf16>, tensor<1xi64>)
    return %4#0, %4#1 : tensor<1xf16>, tensor<1xi64>
  }
}

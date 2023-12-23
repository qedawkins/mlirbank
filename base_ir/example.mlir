func.func @dispatch_3011(%4: tensor<640x5760xf16>, %5: tensor<2x5760x16384xf16>, %6: tensor<640xf16>) -> tensor<2x640x16384xf16> {
  %cst = arith.constant 0.0 : f16
  %7 = tensor.empty() : tensor<2x640x16384xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%7 : tensor<2x640x16384xf16>) -> tensor<2x640x16384xf16>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], 
                      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%4, %5 : tensor<640x5760xf16>, tensor<2x5760x16384xf16>)
            outs(%8 : tensor<2x640x16384xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.mulf %in, %in_0 : f16
          %12 = arith.addf %11, %out : f16
          linalg.yield %12 : f16
        } -> tensor<2x640x16384xf16>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                        iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%9, %6 : tensor<2x640x16384xf16>, tensor<640xf16>)
            outs(%7 : tensor<2x640x16384xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.addf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<2x640x16384xf16>
  return %10 : tensor<2x640x16384xf16>
}

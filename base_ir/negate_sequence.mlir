#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @forward(%arg0 : tensor<1x32x1x128xf16>) -> tensor<1x32x1x128xf16> {
    %159 = tensor.empty() : tensor<1x32x1x128xf16>
    %165 = tensor.empty() : tensor<32x64xf16>
    %extracted_slice_1616 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf16> to tensor<32x64xf16>
    %extracted_slice_1617 = tensor.extract_slice %arg0[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<1x32x1x128xf16> to tensor<32x64xf16>
    %1794 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice_1617 : tensor<32x64xf16>) outs(%165 : tensor<32x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %1906 = arith.negf %in : f16
      linalg.yield %1906 : f16
    } -> tensor<32x64xf16>
    %inserted_slice_1618 = tensor.insert_slice %1794 into %159[0, 0, 0, 0] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<32x64xf16> into tensor<1x32x1x128xf16>
    %inserted_slice_1619 = tensor.insert_slice %extracted_slice_1616 into %inserted_slice_1618[0, 0, 0, 64] [1, 32, 1, 64] [1, 1, 1, 1] : tensor<32x64xf16> into tensor<1x32x1x128xf16>
    return %inserted_slice_1619 : tensor<1x32x1x128xf16>
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module @state_update {
  util.global private @model.layers.0.self_attn.q_proj.weight {noinline} = #stream.parameter.named<"model"::"model.layers.0.self_attn.q_proj.weight"> : tensor<4096x2048xi8>
  func.func @run_forward(%expanded : tensor<1x32x128xf32>, %arg0: tensor<1x1xi64>, %342 : tensor<1x4096xf32>, %model.layers.0.self_attn.q_proj.weight.quant.scale : tensor<4096x32xf32>, %model.layers.0.self_attn.q_proj.weight.quant.zero_point : tensor<4096x32xf32>) -> tensor<1x4096xf32> {
    %model.layers.0.self_attn.q_proj.weight = util.global.load @model.layers.0.self_attn.q_proj.weight : tensor<4096x2048xi8>
    %338 = flow.tensor.bitcast %model.layers.0.self_attn.q_proj.weight : tensor<4096x2048xi8> -> tensor<4096x4096xi4>
    %expanded_190 = tensor.expand_shape %338 [[0], [1, 2]] : tensor<4096x4096xi4> into tensor<4096x32x128xi4>
    %339 = tensor.empty() : tensor<4096x32x128xf32>
    %340 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_190, %model.layers.0.self_attn.q_proj.weight.quant.scale, %model.layers.0.self_attn.q_proj.weight.quant.zero_point : tensor<4096x32x128xi4>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%339 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i4, %in_766: f32, %in_767: f32, %out: f32):
      %6202 = arith.extui %in : i4 to i32
      %6203 = arith.uitofp %6202 : i32 to f32
      %6204 = arith.subf %6203, %in_767 : f32
      %6205 = arith.mulf %6204, %in_766 : f32
      linalg.yield %6205 : f32
    } -> tensor<4096x32x128xf32>
    %343 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%expanded, %340 : tensor<1x32x128xf32>, tensor<4096x32x128xf32>) outs(%342 : tensor<1x4096xf32>) {
    ^bb0(%in: f32, %in_766: f32, %out: f32):
      %6202 = arith.mulf %in, %in_766 : f32
      %6203 = arith.addf %6202, %out : f32
      linalg.yield %6203 : f32
    } -> tensor<1x4096xf32>
    return %343 : tensor<1x4096xf32>
  }
}

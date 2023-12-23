module @state_update {
  util.global private @_params.model.layers.0.self_attn.q_proj.weight {noinline} : tensor<4096x4096xf32>
  func.func private @initialize(%arg0: !torch.vtensor<[?,4096],f32>) -> (!torch.vtensor<[?,4096],f32>) {
    %_params.model.layers.0.self_attn.q_proj.weight = util.global.load @_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32>
    %55 = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.weight : tensor<4096x4096xf32> -> !torch.vtensor<[4096,4096],f32>
    %int0_74 = torch.constant.int 0
    %int1_75 = torch.constant.int 1
    %56 = torch.aten.transpose.int %55, %int0_74, %int1_75 : !torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int -> !torch.vtensor<[4096,4096],f32>
    %59 = torch.aten.mm %arg0, %56 : !torch.vtensor<[?,4096],f32>, !torch.vtensor<[4096,4096],f32> -> !torch.vtensor<[?,4096],f32>
    return %59 : !torch.vtensor<[?,4096],f32>
  }

  // Becomes ->

  util.global private @_params.model.layers.0.self_attn.q_proj.quant.weight {noinline} : tensor<4096x2048xi8>
  util.global private @_params.model.layers.0.self_attn.q_proj.quant.scale {noinline} : tensor<4096x32xf32>
  util.global private @_params.model.layers.0.self_attn.q_proj.quant.zero_point {noinline} : tensor<4096x32xf32>
  func.func private @quant_initialize(%arg0: !torch.vtensor<[?,4096],f32>) -> (!torch.vtensor<[?,4096],f32>) {

    // Load the quantized weights + scales and zero points.
    %_params.model.layers.0.self_attn.q_proj.quant.weight = util.global.load @_params.model.layers.0.self_attn.q_proj.quant.weight : tensor<4096x2048xi8>
    %_params.model.layers.0.self_attn.q_proj.quant.scale = util.global.load @_params.model.layers.0.self_attn.q_proj.quant.scale : tensor<4096x32xf32>
    %_params.model.layers.0.self_attn.q_proj.quant.zero_point = util.global.load @_params.model.layers.0.self_attn.q_proj.quant.zero_point : tensor<4096x32xf32>
    %qweight = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.quant.weight : tensor<4096x2048xi8> -> !torch.vtensor<[4096,2048],ui8>
    %qscale = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.quant.scale : tensor<4096x32xf32> -> !torch.vtensor<[4096,32],f32>
    %qzp = torch_c.from_builtin_tensor %_params.model.layers.0.self_attn.q_proj.quant.zero_point : tensor<4096x32xf32> -> !torch.vtensor<[4096,32],f32>

    %int4 = torch.constant.int 4
    %int128 = torch.constant.int 128
    // Note this will currently fail with the existing lowering due to a missing
    // batch dimension on the LHS. It's likely worth letting the batch dimension
    // be optional and updating the lowering accordingly. We can do a similar
    // thing with the unit dimension on the scales and zero points (infer the
    // broadcast to do based on the rank).
    %59 = torch.operator "quant.matmul_rhs_group_quant"(%arg0, %qweight, %qscale, %qzp, %int4, %int128)
                         : (!torch.vtensor<[?,4096],f32>, !torch.vtensor<[4096,2048],ui8>,
                            !torch.vtensor<[4096,32],f32>, !torch.vtensor<[4096,32],f32>,
                            !torch.int, !torch.int) -> !torch.vtensor<[?,4096],f32>
    return %59 : !torch.vtensor<[?,4096],f32>
  }
}

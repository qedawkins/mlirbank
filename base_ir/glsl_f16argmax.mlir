#spirv_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>}>

module {

  hal.executable.source private @subgroup_argmax attributes {
    objects = #hal.executable.objects<{
      #spirv_target = [
        #hal.executable.object<{
          path = "/home/quinn/nod/uVkCompute/benchmarks/argmax/one_workgroup_argmax_subgroup_f16.spv"
        }>
      ]
    }>
  } {
    // Similar to the above but in-place by using a read/write binding.
    hal.executable.export public @main ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer>
          ]>
        ]>) {
    ^bb0(%device: !hal.device, %workload: index):
      // %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
  }  // hal.executable

  func.func @forward(%arg0: tensor<1x32000xf16>) -> tensor<1xi64> {
    %empty = tensor.empty() : tensor<1xi64>
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x32000xf16>
    %dim_i32 = arith.index_cast %dim : index to i32
    %4 = flow.dispatch @subgroup_argmax::@main[%dim](%dim_i32, %arg0, %empty) : (i32, tensor<1x32000xf16>, tensor<1xi64>) -> %empty
    return %4 : tensor<1xi64>
  }
}

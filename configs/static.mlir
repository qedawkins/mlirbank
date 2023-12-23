hal.executable public @forward_dispatch_0 {
  hal.executable.variant public @vulkan_spirv_fb target(<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>}>) {
    hal.executable.export public @forward_dispatch_0_generic_8x256_f32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<SPIRVSubgroupReduce>, workgroup_size = [64 : index, 1 : index, 1 : index]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @forward_dispatch_0_generic_8x256_f32() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 2.000000e+00 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x256xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x256xf32>> -> tensor<8x256xf32>
        %3 = tensor.empty() : tensor<8xf32>
        %4 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 256]]>} ins(%cst_0 : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x256xf32>) outs(%4 : tensor<8xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 256]]>} {
        ^bb0(%in: f32, %out: f32):
          %6 = math.powf %in, %cst : f32
          %7 = arith.addf %6, %out : f32
          linalg.yield %7 : f32
        } -> tensor<8xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
        return
      }
    }
  }
}

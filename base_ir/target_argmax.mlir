#compilation = #iree_codegen.compilation_info<
    lowering_config  = <tile_sizes = [[], [64]]>,
    translation_info = <SPIRVSubgroupReduce>,
    workgroup_size = [64, 1, 1], subgroup_size = 64> 

module attributes {hal.device.targets = [#hal.device.target<"vulkan", {executable_targets = [#hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>}>], legacy_sync}>]} {
  hal.executable private @forward_dispatch_0 {
    hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>}> {
      hal.executable.export public @forward_dispatch_0_generic_32000_f16xf16xi64 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @forward_dispatch_0_generic_32000_f16xf16xi64() {
          %c0 = arith.constant 0 : index
          %c0_i64 = arith.constant 0 : i64
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0xFC00 : f16
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32000xf16>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f16>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c64) : !flow.dispatch.tensor<writeonly:tensor<i64>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [32000], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32000xf16>> -> tensor<32000xf16>
          %4 = tensor.empty() : tensor<i64>
          %5 = tensor.empty() : tensor<f16>
          %6 = linalg.fill ins(%c0_i64 : i64) outs(%4 : tensor<i64>) -> tensor<i64>
          %7 = linalg.fill ins(%cst : f16) outs(%5 : tensor<f16>) -> tensor<f16>
          %8:2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%3 : tensor<32000xf16>) outs(%7, %6 : tensor<f16>, tensor<i64>) attrs = {compilation_info = #compilation} {
          ^bb0(%in: f16, %out: f16, %out_0: i64):
            %9 = linalg.index 0 : index
            %10 = arith.index_cast %9 : index to i64
            %11 = arith.maximumf %in, %out : f16
            %12 = arith.cmpf ogt, %in, %out : f16
            %13 = arith.select %12, %10, %out_0 : i64
            linalg.yield %11, %13 : f16, i64
          } -> (tensor<f16>, tensor<i64>)
          flow.dispatch.tensor.store %8#0, %1, offsets = [], sizes = [], strides = [] : tensor<f16> -> !flow.dispatch.tensor<writeonly:tensor<f16>>
          flow.dispatch.tensor.store %8#1, %2, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
          return
        }
      }
    }
  }
  func.func @forward(%arg0: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c64000 = arith.constant 64000 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c553648144_i32 = arith.constant 553648144 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c32000 = arith.constant 32000 : index
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input 0") shape([%c1, %c32000]) type(%c553648144_i32) encoding(%c1_i32)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1x32000xf16> in !stream.resource<external>{%c64000}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c128} => !stream.timepoint
    %1 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg1: !stream.resource<external>{%c64000}, %result as %arg2: !stream.resource<external>{%c128}) {
      stream.cmd.dispatch @forward_dispatch_0::@vulkan_spirv_fb::@forward_dispatch_0_generic_32000_f16xf16xi64 {
        ro %arg1[%c0 for %c64000] : !stream.resource<external>{%c64000},
        wo %arg2[%c0 for %c128] : !stream.resource<external>{%c128},
        wo %arg2[%c0 for %c128] : !stream.resource<external>{%c128}
      } attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]}
    } => !stream.timepoint
    %2 = stream.timepoint.await %1 => %result : !stream.resource<external>{%c128}
    %3 = stream.resource.subview %2[%c0] : !stream.resource<external>{%c128} -> !stream.resource<external>{%c2}
    %4 = stream.resource.subview %2[%c64] : !stream.resource<external>{%c128} -> !stream.resource<external>{%c8}
    %5 = stream.tensor.export %3 : tensor<1xf16> in !stream.resource<external>{%c2} -> !hal.buffer_view
    %6 = stream.tensor.export %4 : tensor<1xi64> in !stream.resource<external>{%c8} -> !hal.buffer_view
    return %5, %6 : !hal.buffer_view, !hal.buffer_view
  }
}

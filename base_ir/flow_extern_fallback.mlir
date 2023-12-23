#executable_target_vulkan_spirv_fb = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.6, [Shader, Float64, Float16, Int64, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer, DotProduct, DotProductInputAll, DotProductInput4x8BitPacked, DotProductInput4x8Bit, CooperativeMatrixKHR], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_integer_dot_product, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers, SPV_KHR_cooperative_matrix]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_shared_memory_size = 65536, max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [1024, 1024, 1024], subgroup_size = 64, min_subgroup_size = 32, max_subgroup_size = 64, cooperative_matrix_properties_khr = [#spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = i8, b_type = i8, c_type = i32, result_type = i32, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f16, result_type = f16, acc_sat = false, scope = <Subgroup>>, #spirv.coop_matrix_props_khr<m_size = 16, n_size = 16, k_size = 16, a_type = f16, b_type = f16, c_type = f32, result_type = f32, acc_sat = false, scope = <Subgroup>>]>>}>
#executable_target_vulkan_spirv_fb1 = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformShuffle, GroupNonUniformShuffleRelative], [SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, AMD:DiscreteGPU, #spirv.resource_limits<max_compute_workgroup_invocations = 1024, max_compute_workgroup_size = [128, 8, 4], subgroup_size = 4, cooperative_matrix_properties_khr = []>>}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#device_target_vulkan = #hal.device.target<"vulkan", {executable_targets = [#executable_target_vulkan_spirv_fb, #executable_target_vulkan_spirv_fb1], legacy_sync}>
module attributes {hal.device.targets = [#device_target_vulkan]} {
  flow.executable private @forward_dispatch_0 attributes {hal.executable.objects = #hal.executable.objects<{#executable_target_vulkan_spirv_fb = [#hal.executable.object<{path = "/home/quinn/nod/uVkCompute/benchmarks/argmax/one_workgroup_argmax_subgroup_f16.spv"}>]}>}  {
    flow.executable.export public @main workgroups(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    } attributes {hal.interface.layout = #pipeline_layout}
    flow.executable.export public @forward_dispatch_0 workgroups(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @forward_dispatch_0(%idx: i32, %arg0: !flow.dispatch.tensor<readonly:tensor<32000xf16>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<i64>>) {
        %cst = arith.constant 0xFC00 : f16
        %c0_i64 = arith.constant 0 : i64
        %0 = flow.dispatch.tensor.load %arg0, offsets = [0], sizes = [32000], strides = [1] : !flow.dispatch.tensor<readonly:tensor<32000xf16>> -> tensor<32000xf16>
        %1 = tensor.empty() : tensor<i64>
        %2 = tensor.empty() : tensor<f16>
        %3 = linalg.fill ins(%c0_i64 : i64) outs(%1 : tensor<i64>) -> tensor<i64>
        %4 = linalg.fill ins(%cst : f16) outs(%2 : tensor<f16>) -> tensor<f16>
        %5:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]} ins(%0 : tensor<32000xf16>) outs(%4, %3 : tensor<f16>, tensor<i64>) {
        ^bb0(%in: f16, %out: f16, %out_0: i64):
          %6 = linalg.index 0 : index
          %7 = arith.index_cast %6 : index to i64
          %8 = arith.maximumf %in, %out : f16
          %9 = arith.cmpf ogt, %in, %out : f16
          %10 = arith.select %9, %7, %out_0 : i64
          linalg.yield %8, %10 : f16, i64
        } -> (tensor<f16>, tensor<i64>)
        flow.dispatch.tensor.store %5#1, %arg1, offsets = [], sizes = [], strides = [] : tensor<i64> -> !flow.dispatch.tensor<writeonly:tensor<i64>>
        return
      }
    }
  }
  func.func @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c32000 = arith.constant 32000 : index
    %c32000_i32 = arith.constant 32000 : i32
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<1x32000xf16>
    %1 = flow.tensor.reshape %0 : tensor<1x32000xf16> -> tensor<32000xf16>
    %2 = flow.dispatch {@forward_dispatch_0::@main, @forward_dispatch_0::@forward_dispatch_0}[%c32000](%c32000_i32, %1) : (i32, tensor<32000xf16>) -> tensor<i64>
    %3 = flow.tensor.reshape %2 : tensor<i64> -> tensor<1xi64>
    %4 = hal.tensor.export %3 "output 0" : tensor<1xi64> -> !hal.buffer_view
    return %4 : !hal.buffer_view
  }
}

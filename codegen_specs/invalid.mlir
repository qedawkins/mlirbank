module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    %0:4 = transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %first, %rest = transform.iree.take_first %0#3, %0#2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %first   num_threads [] tile_sizes [4, 16](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %1 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %1 : !transform.any_op
    transform.iree.apply_cse %1 : !transform.any_op
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %rest into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %first_0, %rest_1 = transform.iree.take_first %fused_op, %tiled_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %0#1 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %2 : !transform.any_op
    transform.iree.apply_cse %2 : !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile %first_0[0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %3 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %3 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %3 : !transform.any_op
    transform.iree.apply_cse %3 : !transform.any_op
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %tiled_linalg_op   num_threads [1, 16, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %4 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %4 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %4 : !transform.any_op
    transform.iree.apply_cse %4 : !transform.any_op
    %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %fused_op_4   num_threads [1, 16, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %5 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %5 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %5 : !transform.any_op
    transform.iree.apply_cse %5 : !transform.any_op
    %forall_op_10, %tiled_op_11 = transform.structured.tile_to_forall_op %rest_1   num_threads [1, 16, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %6 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %6 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %6 : !transform.any_op
    transform.iree.apply_cse %6 : !transform.any_op
    apply_patterns to %6 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %7 = transform.structured.vectorize %2 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %7 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %7 : !transform.any_op
    transform.iree.apply_cse %7 : !transform.any_op
    %8 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %8 : !transform.any_op
    transform.iree.apply_cse %8 : !transform.any_op
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %9 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    %10 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %10 : (!transform.any_op) -> ()
    %11 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %11 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %11 workgroup_dims = [64, 1, 1] warp_dims = [2, 1, 1] subgroup_size = 32 : (!transform.any_op) -> ()
    %12 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.any_op
    //%12 = transform.iree.eliminate_gpu_barriers %11 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %12 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %12 : !transform.any_op
    transform.iree.apply_cse %12 : !transform.any_op
    transform.iree.hoist_static_alloc %12 : (!transform.any_op) -> ()
    apply_patterns to %12 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %12 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    apply_patterns to %12 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %12 : !transform.any_op
    transform.iree.apply_cse %12 : !transform.any_op
    //%13 = transform.structured.hoist_redundant_vector_transfers %12 : (!transform.any_op) -> !transform.any_op
    %13 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %13 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %13 : !transform.any_op
    transform.iree.apply_cse %13 : !transform.any_op
    transform.iree.apply_buffer_optimizations %13 : (!transform.any_op) -> ()
    //%14 = transform.iree.eliminate_gpu_barriers %13 : (!transform.any_op) -> !transform.any_op
    %14 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %14 {
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    apply_patterns to %14 {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.any_op
    apply_patterns to %14 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %14 : !transform.any_op
    transform.iree.apply_cse %14 : !transform.any_op
  }
}

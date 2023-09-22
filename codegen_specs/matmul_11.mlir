module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    %0:3 = transform.iree.match_callback failures(propagate) "contraction"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %first, %rest = transform.iree.take_first %0#2, %0#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %first   num_threads [] tile_sizes [1, 64](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
    %tiled_linalg_op, %loops = transform.structured.tile %first_0[0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %padded, %pad = transform.structured.pad %tiled_linalg_op {copy_back = false, pack_paddings = [1, 1, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %2 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %2 : !transform.any_op
    transform.iree.apply_cse %2 : !transform.any_op
    %3 = get_producer_of_operand %padded[2] : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.rewrite_in_destination_passing_style %3 : (!transform.any_op) -> !transform.any_op
    %5 = get_producer_of_operand %padded[0] : (!transform.any_op) -> !transform.any_op
    %6 = get_producer_of_operand %padded[1] : (!transform.any_op) -> !transform.any_op
    %7 = transform.structured.rewrite_in_destination_passing_style %5 : (!transform.any_op) -> !transform.any_op
    %8 = transform.structured.rewrite_in_destination_passing_style %6 : (!transform.any_op) -> !transform.any_op
    %forall_op_4, %tiled_op_5 = transform.structured.tile_to_forall_op %7   num_threads [1, 64, 1] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %9 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %9 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %9 : !transform.any_op
    transform.iree.apply_cse %9 : !transform.any_op
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %8   num_threads [0, 0, 16, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %10 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %10 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %10 : !transform.any_op
    transform.iree.apply_cse %10 : !transform.any_op
    %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %padded   num_threads [0, 2] tile_sizes [](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %11 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %11 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %11 : !transform.any_op
    transform.iree.apply_cse %11 : !transform.any_op
    %forall_op_10, %tiled_op_11 = transform.structured.tile_to_forall_op %fused_op_2   num_threads [0, 2] tile_sizes [](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %12 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %12 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %12 : !transform.any_op
    transform.iree.apply_cse %12 : !transform.any_op
    %forall_op_12, %tiled_op_13 = transform.structured.tile_to_forall_op %rest_1   num_threads [1, 32, 2] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %13 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %13 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %13 : !transform.any_op
    transform.iree.apply_cse %13 : !transform.any_op
    apply_patterns to %2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %14 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %14 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %14 : !transform.any_op
    transform.iree.apply_cse %14 : !transform.any_op
    %15 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %15 {
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op
    %16 = transform.structured.vectorize %15 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %16 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %16 : !transform.any_op
    transform.iree.apply_cse %16 : !transform.any_op
    %17 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %17 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %17 : !transform.any_op
    transform.iree.apply_cse %17 : !transform.any_op
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %18 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    %19 = transform.structured.match ops{["func.func"]} in %18 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %19 : (!transform.any_op) -> ()
    %20 = transform.structured.match ops{["func.func"]} in %18 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %20 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %20 workgroup_dims = [64, 1, 1] warp_dims = [2, 1, 1] subgroup_size = 32 : (!transform.any_op) -> ()
    %21 = transform.iree.eliminate_gpu_barriers %20 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %21 : !transform.any_op
    transform.iree.apply_cse %21 : !transform.any_op
    transform.iree.hoist_static_alloc %21 : (!transform.any_op) -> ()
    apply_patterns to %21 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %21 : !transform.any_op
    transform.iree.apply_cse %21 : !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
    } : !transform.any_op
    %22 = transform.structured.match ops{["scf.for"]} in %21 : (!transform.any_op) -> !transform.op<"scf.for">
    transform.iree.synchronize_loop %22 : (!transform.op<"scf.for">) -> ()
    %23 = transform.structured.hoist_redundant_vector_transfers %21 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %23 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %23 : !transform.any_op
    transform.iree.apply_cse %23 : !transform.any_op
    transform.iree.apply_buffer_optimizations %23 : (!transform.any_op) -> ()
    transform.iree.vector.vector_to_mma_conversion %23 {use_wmma} : (!transform.any_op) -> ()
    apply_patterns to %23 {
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    apply_patterns to %23 {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.any_op
    apply_patterns to %23 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %23 : !transform.any_op
    transform.iree.apply_cse %23 : !transform.any_op
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.iree.apply_cse %func : !transform.any_op
    transform.yield
  }

  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    //===------------------------------------------------------===
    // Step 1. Match relevant ops.
    //===------------------------------------------------------===
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %conv, %trailing, %pooling = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %conv_pad = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 2. Tile to workgroups.
    //===------------------------------------------------------===
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %pooling   num_threads [] tile_sizes [1, 32](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %trailing into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %conv into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_5, %new_containing_op_6 = transform.structured.fuse_into_containing_op %conv_pad into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %new_containing_op_6 : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 3. Materialize input pad.
    //===------------------------------------------------------===
    %padded, %pad = transform.structured.pad %fused_op_2 {copy_back = false, pack_paddings = [1, 0, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %3 = get_producer_of_operand %padded[0] : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.rewrite_in_destination_passing_style %3 : (!transform.any_op) -> !transform.any_op
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %4   num_threads [4, 11] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 4. Tile filter loop
    //===------------------------------------------------------===
    %tiled_linalg_op, %loops:2 = transform.structured.tile %padded[0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 5. Distribute to warps/threads.
    //===------------------------------------------------------===
    %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %tiled_linalg_op   num_threads [2] tile_sizes [](mapping = [#gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %8 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill1, %fill2 = transform.split_handle %8 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %forall_op_10, %tiled_op_11 = transform.structured.tile_to_forall_op %fill1   num_threads [2, 8, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    //%forall_op_12, %tiled_op_13 = transform.structured.tile_to_forall_op %fused_op   num_threads [2, 8, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %forall_op_13, %tiled_op_14 = transform.structured.tile_to_forall_op %tiled_op   num_threads [1, 32, 2] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_7, %new_containing_op_7 = transform.structured.fuse_into_containing_op %fill2 into %forall_op_13 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op_8, %new_containing_op_8 = transform.structured.fuse_into_containing_op %fused_op into %forall_op_13 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 6. Fuse padding with consumers.
    //===------------------------------------------------------===
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %func {
      transform.apply_patterns.iree.swap_tensor_pad
    } : !transform.any_op
    %11 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %func : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_cse %11 : !transform.any_op
    apply_patterns to %11 {
      transform.apply_patterns.iree.swap_tensor_pad
    } : !transform.any_op
    %12 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %11 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_cse %12 : !transform.any_op
    apply_patterns to %12 {
      transform.apply_patterns.iree.swap_tensor_pad
    } : !transform.any_op
    %13 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %12 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_cse %13 : !transform.any_op

    //===------------------------------------------------------===
    // Step 7. Vectorize.
    //===------------------------------------------------------===
    apply_patterns to %13 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %14 = apply_registered_pass "iree-codegen-vectorize-tensor-pad" to %13 : (!transform.any_op) -> !transform.any_op
    %15 = transform.structured.vectorize %14 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op
    %16 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 8. Bufferize.
    //===------------------------------------------------------===
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %17 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    %18 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %18 : (!transform.any_op) -> ()
    %19 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 9. Distribute.
    //===------------------------------------------------------===
    transform.iree.forall_to_workgroup %19 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %19 workgroup_dims = [64, 1, 1] warp_dims = [2, 1, 1] subgroup_size = 32 : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%17) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 10. Cleanup and map to mma.
    //===------------------------------------------------------===
    transform.iree.hoist_static_alloc %19 : (!transform.any_op) -> ()
    apply_patterns to %19 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %19 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%17) : (!transform.any_op) -> ()
    apply_patterns to %19 {
      transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
    } : !transform.any_op
    %20 = transform.structured.match ops{["scf.for"]} in %19 : (!transform.any_op) -> !transform.op<"scf.for">
    transform.iree.synchronize_loop %20 : (!transform.op<"scf.for">) -> ()
    %21 = transform.structured.hoist_redundant_vector_transfers %19 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%17) : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %21 : (!transform.any_op) -> ()
    transform.iree.vector.vector_to_mma_conversion %21 {use_wmma} : (!transform.any_op) -> ()
    apply_patterns to %21 {
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%17) : (!transform.any_op) -> ()
  }
}

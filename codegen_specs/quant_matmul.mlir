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

    // transform.iree.register_match_callbacks
    // %0:3 = transform.iree.match_callback failures(propagate) "contraction"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %dequant, %matmul = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 2. Tile to workgroups.
    //===------------------------------------------------------===
    %forall_op, %tiled_matmul = transform.structured.tile_to_forall_op %matmul   num_threads [] tile_sizes [1, 1, 128](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %fused_dequant, %new_containing_op = transform.structured.fuse_into_containing_op %dequant into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %new_containing_op_2 = transform.structured.fuse_into_containing_op %fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 3. Form the inner loop
    //===------------------------------------------------------===
    %inner_matmul, %loops:2 = transform.structured.tile %tiled_matmul[0, 0, 0, 1, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    //%inner_matmul, %loops:1 = transform.structured.tile_to_scf_for %tiled_matmul[0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_dequant_1, %new_loop_0 = transform.structured.fuse_into_containing_op %fused_dequant into %loops#0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%fused_dequant_2, %new_loop_1 = transform.structured.fuse_into_containing_op %fused_dequant_1 into %loops#1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 4. Materialize linalg copies (later to shared mem)
    //===------------------------------------------------------===
    %padded_matmul, %pad = transform.structured.pad %inner_matmul {copy_back = false, pack_paddings = [1, 0, 1], pad_to_multiple_of = [1, 1, 1, 1, 1], padding_dimensions = [0, 1, 2, 3, 4], padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %3 = get_producer_of_operand %padded_matmul[2] : (!transform.any_op) -> !transform.any_op
    %output_copy = transform.structured.rewrite_in_destination_passing_style %3 : (!transform.any_op) -> !transform.any_op
    %5 = get_producer_of_operand %padded_matmul[0] : (!transform.any_op) -> !transform.any_op
    %input_copy = transform.structured.rewrite_in_destination_passing_style %5 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 5. Tile to threads
    //===------------------------------------------------------===
    %forall_op_4, %tiled_op_5 = transform.structured.tile_to_forall_op %padded_matmul   num_threads [1, 1, 64] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %input_copy   num_threads [1, 1, 1, 32] tile_sizes [](mapping = [#gpu.thread<linear_dim_3>, #gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    //%forall_op_12, %tiled_op_13 = transform.structured.tile_to_forall_op %output_copy   num_threads [1, 1, 64] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %forall_op_13, %tiled_op_14 = transform.structured.tile_to_forall_op %fused_fill   num_threads [1, 1, 64] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %forall_op_14, %tiled_op_15 = transform.structured.tile_to_forall_op %fused_dequant_1   num_threads [64] tile_sizes [](mapping = [#gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 6. Cast away unit dims and vectorize
    //===------------------------------------------------------===
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    %15 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %15 {
      transform.apply_patterns.vector.lower_masked_transfers
    } : !transform.any_op
    %16 = transform.structured.vectorize %15 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 7. Bufferize
    //===------------------------------------------------------===
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %18 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    %19 = transform.structured.match ops{["func.func"]} in %18 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %19 : (!transform.any_op) -> ()
    %20 = transform.structured.match ops{["func.func"]} in %18 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 7. Distribute
    //===------------------------------------------------------===
    transform.iree.forall_to_workgroup %20 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %20 workgroup_dims = [64, 1, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 8. Cleanup and various optimizations
    //===------------------------------------------------------===
    //%21 = transform.iree.eliminate_gpu_barriers %20 : (!transform.any_op) -> !transform.any_op
    %21 = transform.structured.match ops{["func.func"]} in %20 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%18) : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %21 : (!transform.any_op) -> ()
    apply_patterns to %21 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %21 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%18) : (!transform.any_op) -> ()
    // No WMMA for unit M.
    // apply_patterns to %21 {
    //   transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
    // } : !transform.any_op
    %22 = transform.structured.match ops{["scf.for"]} in %21 : (!transform.any_op) -> !transform.op<"scf.for">
    transform.iree.synchronize_loop %22 : (!transform.op<"scf.for">) -> ()
    %23 = transform.structured.hoist_redundant_vector_transfers %21 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%18) : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %23 : (!transform.any_op) -> ()
    // No WMMA for unit M.
    // transform.iree.vector.vector_to_mma_conversion %23 {use_wmma} : (!transform.any_op) -> ()
    apply_patterns to %23 {
      transform.apply_patterns.vector.lower_masks
    } : !transform.any_op
    apply_patterns to %23 {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%18) : (!transform.any_op) -> ()
  }
}

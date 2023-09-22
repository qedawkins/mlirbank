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
    //transform.iree.register_match_callbacks
    //%0:4 = transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    //===------------------------------------------------------===
    // Step 1. Bubble expand_shape to between the conv and trailing.
    //===------------------------------------------------------===
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %func {
      transform.apply_patterns.iree.bubble_expand
    } : !transform.any_op


    //===------------------------------------------------------===
    // Step 2. Match relevant ops.
    //===------------------------------------------------------===
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %conv1, %trailing1, %conv2, %trailing2, %pixel_shuffle = transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %pad = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fills = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fill1, %fill2 = transform.split_handle %fills : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %expand_shape = transform.structured.match ops{["tensor.expand_shape"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 3. Tile to workgroups.
    //===------------------------------------------------------===
    %forall_op, %tiled_pixel_shuffle_op = transform.structured.tile_to_forall_op %pixel_shuffle   num_threads [] tile_sizes [6, 16](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_trailing2, %forall_op_2 = transform.structured.fuse_into_containing_op %trailing2 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_expand_op, %forall_op_3 = transform.structured.fuse_into_containing_op %expand_shape into %forall_op_2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    apply_patterns to %func {
      transform.apply_patterns.iree.swap_tensor_expand_shape
    } : !transform.any_op
    %fused_conv2, %forall_op_4 = transform.structured.fuse_into_containing_op %conv2 into %forall_op_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill2, %forall_op_5 = transform.structured.fuse_into_containing_op %fill2 into %forall_op_4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    apply_patterns to %func {
      transform.apply_patterns.iree.swap_tensor_pad
    } : !transform.any_op
    %fused_trailing1, %forall_op_6 = transform.structured.fuse_into_containing_op %trailing1 into %forall_op_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_conv1, %forall_op_7 = transform.structured.fuse_into_containing_op %conv1 into %forall_op_6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill1, %forall_op_8 = transform.structured.fuse_into_containing_op %fill1 into %forall_op_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op_5 : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 4. Create the input copy.
    //===------------------------------------------------------===
    %padded_conv, %conv_input_pad = transform.structured.pad %fused_conv2 {copy_back = false, pack_paddings = [1, 0, 1], pad_to_multiple_of = [1, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %padded_conv1, %conv1_pad  = transform.structured.pad %fused_conv1 {copy_back = false, pack_paddings = [1, 1, 1], pad_to_multiple_of = [8, 18, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    //%input_pad = get_producer_of_operand %padded_conv[0] : (!transform.any_op) -> !transform.any_op
    //%dps_copy = transform.structured.rewrite_in_destination_passing_style %input_pad : (!transform.any_op) -> !transform.any_op

    ////===------------------------------------------------------===
    //// Step 4. Create the input copy.
    ////===------------------------------------------------------===
    //%forall_op_9, %tiled_dps_copy = transform.structured.tile_to_forall_op %dps_copy   num_threads [8, 6, 1] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //apply_patterns to %func {
    //  transform.apply_patterns.iree.swap_tensor_pad
    //} : !transform.any_op
    //%fused_trailing1_l2, %forall_op_10 = transform.structured.fuse_into_containing_op %fused_trailing1 into %forall_op_9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    ////===------------------------------------------------------===
    //// Step 5. Tile the reductions.
    ////===------------------------------------------------------===
    //%inner_gemm, %loops:2 = transform.structured.tile %padded_conv[0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    //%inner_gemm1, %loop1 = transform.structured.tile %fused_conv1[0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    ////===------------------------------------------------------===
    //// Step 6. Tile to warps (SIMD).
    ////===------------------------------------------------------===
    //%forall_op_11, %tiled_inner_gemm = transform.structured.tile_to_forall_op %inner_gemm   num_threads [2, 1, 1] tile_sizes [](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    ////===------------------------------------------------------===
    //// Step 7. Tile to threads (SIMT).
    ////===------------------------------------------------------===
    //%tiled_fills = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    //%tiled_fill1, %tiled_fill2 = transform.split_handle %tiled_fills : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%forall_op_12, %fill1_l2 = transform.structured.tile_to_forall_op %tiled_fill1   num_threads [8, 1, 8] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%forall_op_13, %fill2_l2 = transform.structured.tile_to_forall_op %tiled_fill2   num_threads [2, 4, 8] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%forall_op_14, %conv1_l3 = transform.structured.tile_to_forall_op %inner_gemm1   num_threads [8, 1, 8] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    //%forall_op_15, %pixel_shuffle_l2 = transform.structured.tile_to_forall_op %tiled_pixel_shuffle_op   num_threads [2, 4, 8] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%trailing_l2, %forall_op_16 = transform.structured.fuse_into_containing_op %fused_trailing2 into %forall_op_15 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    // //===------------------------------------------------------===
    // // Step 8. Fuse padding.
    // //===------------------------------------------------------===
    // apply_patterns to %func {
    //   transform.apply_patterns.iree.swap_tensor_pad
    // } : !transform.any_op
    // %func_1 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %func : (!transform.any_op) -> !transform.any_op
    // transform.iree.apply_cse %func_1 : !transform.any_op
    // apply_patterns to %func_1 {
    //   transform.apply_patterns.iree.swap_tensor_pad
    // } : !transform.any_op
    // %func_2 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %func_1 : (!transform.any_op) -> !transform.any_op
    // transform.iree.apply_cse %func_2 : !transform.any_op
    // apply_patterns to %func_2 {
    //   transform.apply_patterns.iree.swap_tensor_pad
    // } : !transform.any_op
    // %func_3 = apply_registered_pass "iree-codegen-concretize-pad-result-shape" to %func_2 : (!transform.any_op) -> !transform.any_op
    // transform.iree.apply_cse %func_3 : !transform.any_op

    // //===------------------------------------------------------===
    // // Step 9. Cleanup and vectorize.
    // //===------------------------------------------------------===
    // apply_patterns to %func_3 {
    //   transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    //   transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    //   transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    // } : !transform.any_op
    // %func_4 = apply_registered_pass "iree-codegen-vectorize-tensor-pad" to %func_3 : (!transform.any_op) -> !transform.any_op
    // %func_5 = transform.structured.vectorize %func_4 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op
    // transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    // //===------------------------------------------------------===
    // // Step 8. Bufferize.
    // //===------------------------------------------------------===
    // transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    // %17 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    // %18 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op
    // transform.iree.apply_buffer_optimizations %18 : (!transform.any_op) -> ()

    // //===------------------------------------------------------===
    // // Step 9. Distribute to blocks and threads.
    // //===------------------------------------------------------===
    // %19 = transform.structured.match ops{["func.func"]} in %17 : (!transform.any_op) -> !transform.any_op
    // transform.iree.forall_to_workgroup %19 : (!transform.any_op) -> ()
    // transform.iree.map_nested_forall_to_gpu_threads %19 workgroup_dims = [32, 1, 1] warp_dims = [1, 1, 1] subgroup_size = 32 : (!transform.any_op) -> ()
    // apply_patterns to %19 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.iree.fold_fill_into_pad
    //   transform.apply_patterns.scf.for_loop_canonicalization
    //   transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.iree.apply_licm %19 : !transform.any_op
    // transform.iree.apply_cse %19 : !transform.any_op

    // //===------------------------------------------------------===
    // // Step 10. Cleanup + unrolling + optimizations
    // //===------------------------------------------------------===
    // transform.iree.hoist_static_alloc %19 : (!transform.any_op) -> ()
    // apply_patterns to %19 {
    //   transform.apply_patterns.memref.fold_memref_alias_ops
    // } : !transform.any_op
    // apply_patterns to %19 {
    //   transform.apply_patterns.memref.extract_address_computations
    // } : !transform.any_op
    // apply_patterns to %19 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.iree.fold_fill_into_pad
    //   transform.apply_patterns.scf.for_loop_canonicalization
    //   transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.iree.apply_licm %19 : !transform.any_op
    // transform.iree.apply_cse %19 : !transform.any_op
    // apply_patterns to %19 {
    //   transform.apply_patterns.iree.unroll_vectors_gpu_wmma_sync
    // } : !transform.any_op
    // %20 = transform.structured.match ops{["scf.for"]} in %19 : (!transform.any_op) -> !transform.op<"scf.for">
    // transform.iree.synchronize_loop %20 : (!transform.op<"scf.for">) -> ()
    // %21 = transform.structured.hoist_redundant_vector_transfers %19 : (!transform.any_op) -> !transform.any_op
    // apply_patterns to %21 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.iree.fold_fill_into_pad
    //   transform.apply_patterns.scf.for_loop_canonicalization
    //   transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.iree.apply_licm %21 : !transform.any_op
    // transform.iree.apply_cse %21 : !transform.any_op
    // transform.iree.apply_buffer_optimizations %21 : (!transform.any_op) -> ()

    // //===------------------------------------------------------===
    // // Step 11. Map to mma.
    // //===------------------------------------------------------===
    // transform.iree.vector.vector_to_mma_conversion %21 {use_wmma} : (!transform.any_op) -> ()
    // apply_patterns to %21 {
    //   transform.apply_patterns.vector.lower_masks
    // } : !transform.any_op
    // apply_patterns to %21 {
    //   transform.apply_patterns.vector.materialize_masks
    // } : !transform.any_op
    // apply_patterns to %21 {
    //   transform.apply_patterns.linalg.tiling_canonicalization
    //   transform.apply_patterns.iree.fold_fill_into_pad
    //   transform.apply_patterns.scf.for_loop_canonicalization
    //   transform.apply_patterns.canonicalization
    //   transform.apply_patterns.memref.fold_memref_alias_ops
    // } : !transform.any_op
    // transform.iree.apply_licm %21 : !transform.any_op
    // transform.iree.apply_cse %21 : !transform.any_op
  }
}

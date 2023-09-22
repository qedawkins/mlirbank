module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    // Step 1. Match convolution
    %fill, %conv, %trailing = transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // Step 2. Tile to workgroups
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %trailing   num_threads [] tile_sizes [1, 64](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %conv into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
    %3 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %3 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()

    // Step 3. Tile along the filter dimensions.
    %tiled_linalg_op, %loops:2 = transform.structured.tile_to_scf_for %fused_op[0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %forall_op_16, %tiled_op_16 = transform.structured.tile_to_forall_op %tiled_linalg_op   num_threads [0, 1, 1] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    %forall_op_15, %tiled_op_17 = transform.structured.tile_to_forall_op %fused_op_0   num_threads [0, 8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %forall_op_14, %tiled_op_15 = transform.structured.tile_to_forall_op %tiled_op   num_threads [0, 8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %12 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %12 {rank_reducing_linalg, rank_reducing_vector} : (!transform.any_op) -> ()
    %13 = transform.structured.vectorize %12 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %13 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm} : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %14 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    %15 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %15 : (!transform.any_op) -> ()
    %16 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %16 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %16 workgroup_dims = [32, 1, 1] warp_dims = [1, 1, 1] : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %16 : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {fold_memref_aliases} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {extract_address_computations} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.unroll_vectors_gpu_wmma %16 [16, 16, 16] : (!transform.any_op) -> ()
    %17 = transform.structured.hoist_redundant_vector_transfers %16 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %17 : (!transform.any_op) -> ()
    transform.iree.vector.vector_to_mma_conversion %17 {use_wmma} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %17 {fold_memref_aliases} : (!transform.any_op) -> ()





    //%fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %rest into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()
    //%3 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    //%fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %3 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%fused_op_2, %new_containing_op_3 = transform.structured.fuse_into_containing_op %img2col_tensor into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%first_4, %rest_5 = transform.iree.take_first %fused_op, %tiled_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%tiled_linalg_op, %loops = transform.structured.tile_to_scf_for %first_4[0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %fused_op_2 into %loops : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%4 = transform.structured.pad %tiled_linalg_op {pack_paddings = [0, 1, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> !transform.any_op
    //%5 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%6 = get_producer_of_operand %4[2] : (!transform.any_op) -> !transform.any_op
    //%7 = transform.structured.rewrite_in_destination_passing_style %6 : (!transform.any_op) -> !transform.any_op
    //%8 = get_producer_of_operand %4[0] : (!transform.any_op) -> !transform.any_op
    //%9 = get_producer_of_operand %4[1] : (!transform.any_op) -> !transform.any_op
    //%10 = transform.structured.rewrite_in_destination_passing_style %9 : (!transform.any_op) -> !transform.any_op
    //%forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %8   num_threads [32, 1] tile_sizes [](mapping = [#gpu.linear<x>, #gpu.linear<y>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%forall_op_10, %tiled_op_11 = transform.structured.tile_to_forall_op %10   num_threads [16, 2] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%forall_op_12, %tiled_op_13 = transform.structured.tile_to_forall_op %rest_5   num_threads [8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%forall_op_14, %tiled_op_15 = transform.structured.tile_to_forall_op %5   num_threads [8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%forall_op_16, %tiled_op_17 = transform.structured.tile_to_forall_op %4   num_threads [1, 1] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //%11 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    //%12 = transform.vector.lower_masked_transfers %11 : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_patterns %12 {rank_reducing_linalg, rank_reducing_vector} : (!transform.any_op) -> ()
    //%13 = transform.structured.vectorize %12 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_patterns %13 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %arg0 {canonicalization, cse, licm} : (!transform.any_op) -> ()
    //transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    //%14 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op
    //%15 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_buffer_optimizations %15 : (!transform.any_op) -> ()
    //%16 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    //transform.iree.forall_to_workgroup %16 : (!transform.any_op) -> ()
    //transform.iree.map_nested_forall_to_gpu_threads %16 workgroup_dims = [32, 1, 1] warp_dims = [1, 1, 1] : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.hoist_static_alloc %16 : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %16 {fold_memref_aliases} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %16 {extract_address_computations} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.unroll_vectors_gpu_wmma %16 [16, 16, 16] : (!transform.any_op) -> ()
    //%17 = transform.structured.hoist_redundant_vector_transfers %16 : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.apply_buffer_optimizations %17 : (!transform.any_op) -> ()
    //transform.iree.vector.vector_to_mma_conversion %17 {use_wmma} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    //transform.iree.apply_patterns %17 {fold_memref_aliases} : (!transform.any_op) -> ()
    //%18 = transform.structured.match ops{["memref.alloc"]} in %17 : (!transform.any_op) -> !transform.op<"memref.alloc">
    //%19 = transform.memref.multibuffer %18 {factor = 2 : i64, skip_analysis} : (!transform.op<"memref.alloc">) -> !transform.any_op
    //transform.iree.apply_patterns %17 {fold_memref_aliases} : (!transform.any_op) -> ()
    //%20 = transform.structured.match ops{["gpu.subgroup_mma_compute"]} in %17 : (!transform.any_op) -> !transform.any_op
    //%21 = transform.loop.get_parent_for %20 : (!transform.any_op) -> !transform.any_op
    //%22 = transform.iree.pipeline_shared_memory_copies %21 {depth = 1 : i64, peel_epilogue} : (!transform.any_op) -> !transform.any_op
    //transform.iree.apply_patterns %17 {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization} : (!transform.any_op) -> ()
  }
}

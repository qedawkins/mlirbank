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
    //%inner_channel, %loop = transform.structured.tile_to_scf_for %fused_op[0, 0, 0, 0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %padded = transform.structured.pad %fused_op {pack_paddings = [1, 0, 1], padding_dimensions = [0, 1, 2], padding_values = [0 : i8, 0 : i8, 0 : i32]} : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    %5 = get_producer_of_operand %padded[0] : (!transform.any_op) -> !transform.any_op
    %6 = get_producer_of_operand %padded[1] : (!transform.any_op) -> !transform.any_op
    %7 = get_producer_of_operand %padded[2] : (!transform.any_op) -> !transform.any_op
    %inp_copy = transform.structured.rewrite_in_destination_passing_style %5 : (!transform.any_op) -> !transform.any_op
    //%fil_copy = transform.structured.rewrite_in_destination_passing_style %6 : (!transform.any_op) -> !transform.any_op
    //%out_copy = transform.structured.rewrite_in_destination_passing_style %7 : (!transform.any_op) -> !transform.any_op

    // Step 4. Tile along the filter dimensions.
    %tiled_linalg_op, %loops:2 = transform.structured.tile_to_scf_for %padded[0, 0, 0, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %forall_op_15, %tiled_op_15 = transform.structured.tile_to_forall_op %tiled_linalg_op   num_threads [0, 1, 1] tile_sizes [](mapping = [#gpu.warp<y>, #gpu.warp<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    %forall_op_16, %tiled_op_16 = transform.structured.tile_to_forall_op %fused_op_0   num_threads [0, 8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %forall_op_17, %tiled_op_17 = transform.structured.tile_to_forall_op %tiled_op   num_threads [0, 8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %forall_op_18, %tiled_op_18 = transform.structured.tile_to_forall_op %inp_copy   num_threads [3, 6, 0] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%forall_op_19, %tiled_op_19 = transform.structured.tile_to_forall_op %fil_copy   num_threads [3, 3, 2] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //%forall_op_20, %tiled_op_20 = transform.structured.tile_to_forall_op %out_copy   num_threads [0, 8, 4] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 5. Vectorize.
    %12 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %12 {rank_reducing_linalg, rank_reducing_vector} : (!transform.any_op) -> ()
    %13 = transform.structured.vectorize %12 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op

    // Step 6. Bufferize.
    transform.iree.apply_patterns %13 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm} : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %arg0 : (!transform.any_op) -> ()
    %14 = transform.iree.bufferize {target_gpu} %arg0 : (!transform.any_op) -> !transform.any_op

    // Step 7. Map to workgroups.
    %15 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_buffer_optimizations %15 : (!transform.any_op) -> ()
    %16 = transform.structured.match ops{["func.func"]} in %14 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %16 : (!transform.any_op) -> ()

    // // Step 8. Optionally unroll.
    // %for_loops = transform.structured.match ops{["scf.for"]} in %16 : (!transform.any_op) -> !transform.any_op
    // transform.loop.unroll %for_loops {factor = 3} : !transform.any_op

    // Step 9. Map to threads.
    transform.iree.map_nested_forall_to_gpu_threads %16 workgroup_dims = [32, 1, 1] warp_dims = [1, 1, 1] : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %16 : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {fold_memref_aliases} : (!transform.any_op) -> ()

    // Step 10. Cleanup and conversion to mma.
    transform.iree.apply_patterns %16 {extract_address_computations} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %16 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.unroll_vectors_gpu_wmma %16 [16, 16, 16] : (!transform.any_op) -> ()
    %17 = transform.structured.hoist_redundant_vector_transfers %16 : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %17 : (!transform.any_op) -> ()
    transform.iree.vector.vector_to_mma_conversion %17 {use_wmma} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %17 {fold_memref_aliases} : (!transform.any_op) -> ()
    transform.iree.apply_patterns %17 {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
  }
}

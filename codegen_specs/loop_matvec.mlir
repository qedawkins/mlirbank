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

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %eltwise, %reduction = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
    // Step 1. First level of tiling + fusion parallelizes to blocks.
    // ===========================================================================
    %forall_grid, %reduction_grid =
      transform.structured.tile_to_forall_op %reduction tile_sizes [1]
        ( mapping = [#gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    %not_reduction = transform.merge_handles %fill, %eltwise : !transform.any_op
    transform.structured.fuse_into_containing_op %not_reduction into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Step 2. Copy the input.
    // ===========================================================================
    %padded_reduc, %pad = transform.structured.pad %reduction_grid {copy_back_op = "none", pack_paddings = [1, 0, 0], pad_to_multiple_of = [1, 1], padding_dimensions = [2, 3], padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f16]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    %3 = get_producer_of_operand %padded_reduc[0] : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.rewrite_in_destination_passing_style %3 : (!transform.any_op) -> !transform.any_op
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %4   num_threads [32, 2] tile_sizes [](mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Step 3. Tile the reduction dim.
    // ===========================================================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %dequant = get_producer_of_operand %padded_reduc[1] : (!transform.any_op) -> !transform.any_op
    %split_for, %new_fill, %split_reduc, %combiner_op = transform.structured.tile_reduction_using_scf %padded_reduc by tile_sizes = [0, 32, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
       transform.apply_patterns.tensor.reassociative_reshape_folding
       transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.structured.fuse_into_containing_op %dequant into %split_for : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    //// Step 4. Tile the parallel dim.
    //// ===========================================================================
    ////%split_forall, %new_fill, %split_reduc, %combiner_op = transform.structured.tile_reduction_using_forall %padded_reduc by num_threads = [0, 0, 4, 1], tile_sizes = [], mapping = [#gpu.thread<z>, #gpu.thread<y>]  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    //%tiled_dequant = get_producer_of_operand %split_reduc[1] : (!transform.any_op) -> !transform.any_op
    //%outer_forall, %tiled_reduc = transform.structured.tile_to_forall_op %split_reduc num_threads [4, 32, 2] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    ////%fill_3d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<4xf16> in %variant_op : (!transform.any_op) -> !transform.any_op
    ////transform.structured.fuse_into_containing_op %fill_3d into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.structured.fuse_into_containing_op %tiled_dequant into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    //%fill_3d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<4x32x16xf16> in %variant_op : (!transform.any_op) -> !transform.any_op
    //%fill_forall, %tiled_fill = transform.structured.tile_to_forall_op %fill_3d num_threads [4, 32, 2] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    //%fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<4xf16> in %variant_op : (!transform.any_op) -> !transform.any_op
    //%combiner_forall, %tiled_combiner = transform.structured.tile_to_forall_op %combiner_op num_threads [4] tile_sizes [](mapping = [#gpu.thread<y>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.structured.fuse_into_containing_op %fill_1d into %combiner_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    //////// Step 5. Second level of tiling + fusion parallelizes to threads. Also
    //////// fuse in the leading elementwise.
    //////// ===========================================================================
    //////%fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x1x1xf32> in %variant_op : (!transform.any_op) -> !transform.any_op
    //////%forall_block_combiner_op, %block_combiner_op =
    //////  transform.structured.tile_to_forall_op %combiner_op tile_sizes [0, 0, 1]
    //////  ( mapping = [#gpu.thread<z>] )
    //////  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //////transform.structured.fuse_into_containing_op %fill_1d into %forall_block_combiner_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //////transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
  
    //////%fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<1x1x1x2xf32> in %variant_op : (!transform.any_op) -> !transform.any_op
    //////%grid_eltwise_op = transform.structured.match ops{["linalg.generic"]}
    //////  attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //////transform.structured.fuse_into_containing_op %fill_2d into %split_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //////transform.structured.fuse_into_containing_op %grid_eltwise_op into %split_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //////transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
  
    //// Step 6. Rank-reduce and vectorize.
    //// ===========================================================================
    //%func_1 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //transform.apply_patterns to %func_1 {
    //  transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    //  transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    //  transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    //} : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //// %func_3 = transform.structured.vectorize %func_1 : (!transform.any_op) -> !transform.any_op
    ////%func_3 = apply_registered_pass "iree-codegen-gpu-vectorization" to %func_1 {options = "generate-contract=false"} : (!transform.any_op) -> !transform.any_op
    ////%func_3 = transform.structured.vectorize %func_1 {disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns} : (!transform.any_op) -> !transform.any_op
    //%func_3 = transform.structured.vectorize %func_1 {disable_transfer_permutation_map_lowering_patterns} : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
  
    //// Step 7. Bufferize and drop HAL decriptor from memref ops.
    //// ===========================================================================
    //transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    ////%variant_op_2 = transform.iree.bufferize { target_gpu, test_analysis_only, print_conflicts } %variant_op : (!transform.any_op) -> !transform.any_op
    //%variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
    //%func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //transform.apply_patterns to %func_2 {
    //   transform.apply_patterns.memref.fold_memref_alias_ops
    //} : !transform.any_op
    //transform.iree.apply_buffer_optimizations %func_2 : (!transform.any_op) -> ()
    //transform.iree.apply_cse %func_2 : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
  
    //// Step 6. Post-bufferization mapping to blocks and threads.
    //// ===========================================================================
    //%func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //transform.iree.forall_to_workgroup %func_4 : (!transform.any_op) -> ()
    //transform.iree.map_nested_forall_to_gpu_threads %func_4 workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()
  
    //// Step 7. Post-bufferization vector distribution with rank-reduction.
    //// ===========================================================================
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.apply_patterns to %func_4 {
    //  transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    //  transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    //  transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    //} : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.iree.hoist_static_alloc %func_4 : (!transform.any_op) -> ()
    //apply_patterns to %func_4 {
    //  transform.apply_patterns.memref.fold_memref_alias_ops
    //} : !transform.any_op
    //apply_patterns to %func_4 {
    //  transform.apply_patterns.memref.extract_address_computations
    //} : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //// Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
    //// at this point.
    //transform.sequence %variant_op_2 : !transform.any_op failures(suppress) {
    //^bb0(%arg0: !transform.any_op):
    //  transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 64 }
    //  : (!transform.any_op) -> !transform.any_op
    //}
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.iree.vector.warp_distribute %func_4 : (!transform.any_op) -> ()
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
  }
}

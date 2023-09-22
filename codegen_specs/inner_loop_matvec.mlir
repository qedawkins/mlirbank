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
    //transform.print %variant_op {name = "Before Starting IR: "} : !transform.any_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %eltwise, %reduction = transform.split_handle %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %init_or_alloc_op, %more_parallel_fill_op, %more_parallel_op, %combiner_op =
      transform.structured.split_reduction %reduction
        split_factor = [8, 16] insert_split_dimension = [1, 3] inner_parallel = [true, false]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    //transform.print %variant_op {name = "========= After Splitting the GEMV: "} : !transform.any_op
  
    // Step 1. First level of tiling + fusion parallelizes to blocks.
    // ===========================================================================
    %forall_grid, %grid_combiner_op =
      transform.structured.tile_to_forall_op %combiner_op tile_sizes [8]
        ( mapping = [#gpu.block<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Tiling to Workgroups: "} : !transform.any_op
    //%not_reduction = transform.merge_handles %fill, %eltwise : !transform.any_op
    //transform.structured.fuse_into_containing_op %not_reduction into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Step 2.1: Cannot fuse across the "expand_shape" produced by reduction
    // splitting above, so we need to bubble that up via patterns and rematch
    // the entire structure.
    // TODO: bubbling should be a proper transform op, at which point we will be
    // able to preserve the handles.
    // ===========================================================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.bubble_expand
    } : !transform.any_op
    //transform.print %variant_op {name = "========= After Bubbling tensor.expand_shape ops: "} : !transform.any_op
    %fills = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill_2, %more_parallel_fill_2 = transform.split_handle %fills
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %generics = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %expanded_eltwise, %more_parallel_2, %combiner_2 =
      transform.split_handle %generics : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %forall_grid_2 = transform.structured.match ops{["scf.forall"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %not_combiner = transform.merge_handles %fill_2, %more_parallel_fill_2, %more_parallel_2, %expanded_eltwise : !transform.any_op
    transform.structured.fuse_into_containing_op %not_combiner into %forall_grid_2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.bubble_expand
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Fusing into workgroup level forall and bubbling expand again: "} : !transform.any_op

    // Step 2. Copy the input.
    // ===========================================================================
    %grid_more_parallel_op = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %padded_reduc, %pad = transform.structured.pad %grid_more_parallel_op {copy_back_op = "none", pack_paddings = [1, 0, 0], pad_to_multiple_of = [1, 1, 1, 1], padding_dimensions = [0, 1, 2, 3], padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f16]} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.print %variant_op {name = "========= After Padding LHS to generate copy: "} : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    %3 = get_producer_of_operand %padded_reduc[0] : (!transform.any_op) -> !transform.any_op
    %4 = transform.structured.rewrite_in_destination_passing_style %3 : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op {name = "========= After rewriting pad in DPS: "} : !transform.any_op
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %4   num_threads [8, 8, 16, 1] tile_sizes [](mapping = [#gpu.thread<linear_dim_3>, #gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Tiling copy to threads: "} : !transform.any_op

    // Step 4. Tile the parallel dim.
    // ===========================================================================
    %outer_forall, %tiled_reduc = transform.structured.tile_to_forall_op %combiner_2 num_threads [8] tile_sizes [](mapping = [#gpu.thread<y>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    %fill_1d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<8xf16> in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.fuse_into_containing_op %fill_1d into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Tiling combiner + fuse fill: "} : !transform.any_op

    // Step 3. Tile the reduction dim.
    // ===========================================================================
    %dequant = get_producer_of_operand %padded_reduc[1] : (!transform.any_op) -> !transform.any_op
    %loop_reduc, %new_for = transform.structured.tile_to_scf_for %padded_reduc [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
       transform.apply_patterns.tensor.reassociative_reshape_folding
       transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.structured.fuse_into_containing_op %dequant into %new_for : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Tiling reduction loop of partial reduction generic + fuse dequant: "} : !transform.any_op

    // Step 4. Tile the parallel dim.
    // ===========================================================================
    %tiled_dequant = get_producer_of_operand %loop_reduc[1] : (!transform.any_op) -> !transform.any_op
    %partial_forall, %partial_reduc = transform.structured.tile_to_forall_op %loop_reduc num_threads [8, 8, 0, 16] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.structured.fuse_into_containing_op %tiled_dequant into %partial_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Tiling + distribute of the partial reduction to all threads: "} : !transform.any_op

    %fill_2d = transform.structured.match ops{["linalg.fill"]} filter_result_type = tensor<8x8x16xf16> in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill_forall, %tiled_fill = transform.structured.tile_to_forall_op %fill_2d num_threads [8, 8, 16] tile_sizes [](mapping = [#gpu.thread<linear_dim_2>, #gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>])  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.print %variant_op {name = "========= After Tiling the partial reduction's fill to all threads: "} : !transform.any_op

    // Step 6. Rank-reduce and vectorize.
    // ===========================================================================
    %func_1 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_1 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    // %func_3 = transform.structured.vectorize %func_1 : (!transform.any_op) -> !transform.any_op
    //%func_3 = apply_registered_pass "iree-codegen-gpu-vectorization" to %func_1 {options = "generate-contract=false"} : (!transform.any_op) -> !transform.any_op
    //%func_3 = transform.structured.vectorize %func_1 {disable_multi_reduction_to_contract_patterns, disable_transfer_permutation_map_lowering_patterns} : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op {name = "========= Before Vectorization: "} : !transform.any_op
    %func_3 = transform.structured.vectorize %func_1 {disable_transfer_permutation_map_lowering_patterns} : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //transform.print %variant_op {name = "========= After Vectorization: "} : !transform.any_op
  
    // Step 7. Bufferize and drop HAL decriptor from memref ops.
    // ===========================================================================
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    //%variant_op_2 = transform.iree.bufferize { target_gpu, test_analysis_only, print_conflicts } %variant_op : (!transform.any_op) -> !transform.any_op
    %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> !transform.any_op
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Bufferization: "} : !transform.any_op
    transform.apply_patterns to %func_2 {
       transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Folding memref alias ops: "} : !transform.any_op
    transform.iree.apply_buffer_optimizations %func_2 : (!transform.any_op) -> ()
    transform.iree.apply_cse %func_2 : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Applying buffer optimizations: "} : !transform.any_op
  
    // Step 6. Post-bufferization mapping to blocks and threads.
    // ===========================================================================
    %func_m = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_m : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_m workgroup_dims = [128, 8, 1] subgroup_size = 64 : (!transform.any_op) -> ()
    //transform.print %variant_op_2 {name = "========= After mapping to workgroups + threads: "} : !transform.any_op

    %func_n = transform.structured.hoist_redundant_vector_transfers %func_m : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    %func_4 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Hoisting redundant vector transfers: "} : !transform.any_op
  
    // Step 7. Post-bufferization vector distribution with rank-reduction.
    // ===========================================================================
    %vfunc = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vfunc {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %vfunc : !transform.any_op
    transform.iree.apply_cse %vfunc : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.print %variant_op_2 {name = "========= After Canonicalization + CSE + cleanup: "} : !transform.any_op
    transform.apply_patterns to %func_4 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %func_4 : (!transform.any_op) -> ()
    apply_patterns to %func_4 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %func_4 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Canonicalization + CSE + cleanup + hoisting static allocs: "} : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    // Don't complain about unsupported if (threadIdx.x == 0 && threadIdx.y == 0)
    // at this point.
    transform.sequence %variant_op_2 : !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 128 }
      : (!transform.any_op) -> !transform.any_op
    }
    //transform.print %variant_op_2 {name = "========= After Raising scf.if to warp_execute_on_lane_0: "} : !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    transform.iree.vector.warp_distribute %func_4 : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.print %variant_op_2 {name = "========= After Distributing to warps: "} : !transform.any_op
    apply_patterns to %func_4 {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    apply_patterns to %func_4 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    %func_5_3 = apply_registered_pass "iree-codegen-cleanup-buffer-alloc-view" to %func_4 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.print %variant_op_2 {name = "========= After Folding memref aliases: "} : !transform.any_op
    %func_10 = apply_registered_pass "iree-codegen-optimize-vector-transfer" to %func_5_3 : (!transform.any_op) -> !transform.any_op
    transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //transform.print %variant_op_2 {name = "========= After Optimizing vector transfers: "} : !transform.any_op
    %func_7 = transform.iree.eliminate_gpu_barriers %func_10 : (!transform.any_op) -> !transform.any_op
    //transform.print %variant_op_2 {name = "========= After Eliminating barriers: "} : !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_6 = apply_registered_pass "iree-spirv-vectorize" to %func_7 : (!transform.any_op) -> !transform.any_op




    //%func_5 = transform.structured.hoist_redundant_vector_transfers %func_4 : (!transform.any_op) -> !transform.any_op
    //%func_5_2 = apply_registered_pass "resolve-shaped-type-result-dims" to %func_5 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_5_3 = apply_registered_pass "iree-codegen-cleanup-buffer-alloc-view" to %func_5_2 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_6 = apply_registered_pass "iree-spirv-vectorize" to %func_5_3 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_7 = transform.iree.eliminate_gpu_barriers %func_6 : (!transform.any_op) -> !transform.any_op
    //%func_8 = apply_registered_pass "iree-codegen-canonicalize-scf-for" to %func_7 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    ////%module = transform.structured.match ops{["builtin.module"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    ////%module_2 = apply_registered_pass "iree-spirv-vectorize-load-store" to %module : (!transform.any_op) -> !transform.any_op
    ////transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_9 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!transform.any_op) -> !transform.any_op
    //%func_10 = apply_registered_pass "iree-codegen-optimize-vector-transfer" to %func_9 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //apply_patterns to %func_9 {
    //  transform.apply_patterns.memref.fold_memref_alias_ops
    //  transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    //} : !transform.any_op
    //%func_10 = apply_registered_pass "iree-codegen-optimize-vector-transfer" to %func_9 {options = "flatten=true"} : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
    //%func_11 = apply_registered_pass "iree-codegen-optimize-vector-transfer" to %func_10 : (!transform.any_op) -> !transform.any_op
    //transform.include @cleanup failures(propagate) (%variant_op_2) : (!transform.any_op) -> ()
  }
}

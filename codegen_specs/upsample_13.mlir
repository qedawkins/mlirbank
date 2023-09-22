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
    %upsample = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    //===------------------------------------------------------===
    // Step 2. Tile to workgroups.
    //===------------------------------------------------------===
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %upsample   num_threads [] tile_sizes [1, 2, 32](mapping = [#gpu.block<z>, #gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!transform.any_op) -> ()


    //===------------------------------------------------------===
    // Step 3. Distribute to threads.
    //===------------------------------------------------------===
    %forall_op_threads, %tiled_op_threads = transform.structured.tile_to_forall_op %tiled_op   num_threads [0, 2, 32, 1] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<y>, #gpu.linear<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.include @cleanup failures(propagate) (%arg0) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 7. Vectorize.
    //===------------------------------------------------------===
    %13 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    apply_patterns to %13 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %14 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %15 = transform.structured.vectorize %14 {vectorize_nd_extract} : (!transform.any_op) -> !transform.any_op
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
    transform.iree.map_nested_forall_to_gpu_threads %19 workgroup_dims = [64, 1, 1] warp_dims = [1, 1, 1] subgroup_size = 64 : (!transform.any_op) -> ()
    transform.include @cleanup failures(propagate) (%17) : (!transform.any_op) -> ()

    //===------------------------------------------------------===
    // Step 10. Cleanup.
    //===------------------------------------------------------===
    transform.iree.hoist_static_alloc %19 : (!transform.any_op) -> ()
    apply_patterns to %19 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    apply_patterns to %19 {
      transform.apply_patterns.memref.extract_address_computations
    } : !transform.any_op
    transform.include @cleanup failures(propagate) (%19) : (!transform.any_op) -> ()
    transform.iree.apply_buffer_optimizations %19 : (!transform.any_op) -> ()
  }
}

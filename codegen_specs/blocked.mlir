module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.register_match_callbacks
    %0:3 = transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
    %img2col_tensor, %transformed = transform.structured.convert_conv2d_to_img2col %0#1 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
    %1 = get_producer_of_operand %transformed[0] : (!pdl.operation) -> !pdl.operation
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %2 {bubble_collapse} : (!pdl.operation) -> ()
    %first, %rest = transform.iree.take_first %0#2, %1 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %first   num_threads [] tile_sizes [2, 64](mapping = [#gpu.block<y>, #gpu.block<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %3 = transform.structured.fuse_into_containing_op %rest into %forall_op
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!pdl.operation) -> ()
    %4 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %5 = transform.structured.fuse_into_containing_op %4 into %forall_op
    %6 = transform.structured.fuse_into_containing_op %img2col_tensor into %forall_op
    %first_0, %rest_1 = transform.iree.take_first %3, %tiled_op : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
    %tiled_linalg_op, %loops = transform.structured.tile_to_scf_for %first_0[0, 0, 0, 1]
    %7 = transform.structured.fuse_into_containing_op %6 into %loops
    %8 = transform.structured.pad %tiled_linalg_op {pack_paddings = [0, 1, 1], padding_dimensions = [0, 1, 2, 3, 4], padding_values = [0 : i8, 0 : i8, 0 : i32]}
    %9 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %10 = get_producer_of_operand %8[2] : (!pdl.operation) -> !pdl.operation
    %11 = transform.structured.rewrite_in_destination_passing_style %10 : (!pdl.operation) -> !pdl.operation
    %12 = get_producer_of_operand %8[1] : (!pdl.operation) -> !pdl.operation
    %13 = get_producer_of_operand %8[0] : (!pdl.operation) -> !pdl.operation
    %14 = transform.structured.rewrite_in_destination_passing_style %12 : (!pdl.operation) -> !pdl.operation
    %forall_op_2, %tiled_op_3 = transform.structured.tile_to_forall_op %14   num_threads [2, 0, 16, 4] tile_sizes [](mapping = [#gpu.linear<z>, #gpu.linear<x>, #gpu.linear<y>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_4, %tiled_op_5 = transform.structured.tile_to_forall_op %13   num_threads [0, 64, 2] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %rest_1   num_threads [2, 64] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %9   num_threads [2, 64] tile_sizes [](mapping = [#gpu.linear<y>, #gpu.linear<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_10, %tiled_op_11 = transform.structured.tile_to_forall_op %8   num_threads [2, 2, 1] tile_sizes [](mapping = [#gpu.warp<z>, #gpu.warp<y>, #gpu.warp<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %15 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %15 {rank_reducing_linalg, rank_reducing_vector} : (!pdl.operation) -> ()
    %16 = transform.structured.vectorize %15 {vectorize_nd_extract}
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm} : (!pdl.operation) -> ()
    transform.iree.eliminate_empty_tensors %arg0 : (!pdl.operation) -> ()
    %17 = transform.iree.bufferize {target_gpu} %arg0 : (!pdl.operation) -> !pdl.operation
    %18 = transform.structured.match ops{["func.func"]} in %17 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_buffer_optimizations %18 : (!pdl.operation) -> ()
    %19 = transform.structured.match ops{["func.func"]} in %17 : (!pdl.operation) -> !pdl.operation
    transform.iree.forall_to_workgroup %19 : (!pdl.operation) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %19 workgroup_dims = [32, 2, 2] warp_dims = [1, 2, 2] : (!pdl.operation) -> ()
    transform.iree.apply_patterns %19 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.hoist_static_alloc %19 : (!pdl.operation) -> ()
    transform.iree.apply_patterns %19 {fold_memref_aliases} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %19 {extract_address_computations} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %19 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.unroll_vectors_gpu_wmma %19 [16, 16, 16] : (!pdl.operation) -> ()
    %20 = transform.structured.hoist_redundant_vector_transfers %19 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %20 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.apply_buffer_optimizations %20 : (!pdl.operation) -> ()
    transform.iree.vector.vector_to_mma_conversion %20 {use_wmma} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %20 {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization} : (!pdl.operation) -> ()
  }
}

module {
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    print %arg0 : !pdl.operation
    transform.iree.register_match_callbacks
    %0:4 = transform.iree.match_callback failures(propagate) "convolution"(%arg0) : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    %img2col_tensor, %transformed = transform.structured.convert_conv2d_to_img2col %0#2 : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
    %1 = get_producer_of_operand %transformed[0] : (!pdl.operation) -> !pdl.operation
    print %arg0 : !pdl.operation
    %2 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %2 {bubble_collapse} : (!pdl.operation) -> ()
    print %arg0 : !pdl.operation
    %first, %rest = transform.iree.take_first %0#3, %1 : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %first   num_threads [] tile_sizes [32, 16](mapping = [#gpu.block<x>, #gpu.block<y>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %3 = transform.structured.fuse_into_containing_op %rest into %forall_op
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_op : (!pdl.operation) -> ()
    %4 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %5 = transform.structured.fuse_into_containing_op %4 into %forall_op
    %6 = transform.structured.fuse_into_containing_op %img2col_tensor into %forall_op
    print %arg0 : !pdl.operation
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %first_0, %rest_1 = transform.iree.take_first %3, %tiled_op : (!pdl.operation, !pdl.operation) -> (!pdl.operation, !pdl.operation)
    %tiled_linalg_op, %loops = transform.structured.tile_to_scf_for %first_0[0, 0, 16]
    %7 = transform.structured.fuse_into_containing_op %6 into %loops
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    print %arg0 : !pdl.operation
    %8:2 = transform.iree.promote_operands %tiled_linalg_op [1] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
    print %arg0 : !pdl.operation
    %forall_op_2, %tiled_op_3 = transform.structured.tile_to_forall_op %7   num_threads [32] tile_sizes [](mapping = [#gpu.thread<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_4, %tiled_op_5 = transform.structured.tile_to_forall_op %rest_1   num_threads [32] tile_sizes [](mapping = [#gpu.thread<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    %forall_op_6, %tiled_op_7 = transform.structured.tile_to_forall_op %5   num_threads [32] tile_sizes [](mapping = [#gpu.thread<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    print %arg0 : !pdl.operation
    %forall_op_8, %tiled_op_9 = transform.structured.tile_to_forall_op %8#0   num_threads [1] tile_sizes [](mapping = [#gpu.warp<x>])
    transform.iree.apply_patterns %arg0 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    print %arg0 : !pdl.operation
    %9 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %9 {rank_reducing_linalg, rank_reducing_vector} : (!pdl.operation) -> ()
    %10 = transform.structured.vectorize %9 {vectorize_nd_extract}
    print %arg0 : !pdl.operation
    %11 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %12:5 = split_handles %11 in[5] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)
    transform.iree.apply_patterns_to_nested %12#2 {unroll_vectors_gpu_coop_mat} : (!pdl.operation) -> ()
    print %arg0 : !pdl.operation
    transform.iree.apply_patterns %10 {fold_reassociative_reshapes} : (!pdl.operation) -> ()
    transform.iree.eliminate_empty_tensors %arg0 : (!pdl.operation) -> ()
    %13 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %13 {erase_unnecessary_tensor_operands} : (!pdl.operation) -> ()
    %14 = transform.iree.bufferize {target_gpu} %arg0 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    print %14 : !pdl.operation
    %15 = transform.structured.match ops{["func.func"]} in %14 : (!pdl.operation) -> !pdl.operation
    transform.iree.forall_to_workgroup %15 : (!pdl.operation) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %15 workgroup_dims = [32, 1, 1] warp_dims = [1, 1, 1] : (!pdl.operation) -> ()
    transform.iree.hoist_static_alloc %15 : (!pdl.operation) -> ()
    %16 = transform.iree.gpu_distribute_shared_memory_copy %15 : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    transform.iree.vector.vector_to_mma_conversion %16 {use_wmma} : (!pdl.operation) -> ()
    transform.iree.apply_patterns %14 {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
    print %14 : !pdl.operation
  }
}

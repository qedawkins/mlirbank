// in_width = 256
// depth = 11
// out_width = 128
// n_nodes = 2 ** (depth + 1) - 1 = 4095
// batch_size = ?

!w1_t = tensor<4095x768xf32>
!w2_t = tensor<4095x768xf32>
!full_in_t = tensor<?x768xf32>
!in_t = tensor<768xf32>
!score_t = tensor<f32>
!full_out_t = tensor<?x768xf32>
!out_t = tensor<768xf32>

#pipeline_layout = #hal.pipeline.layout<push_constants = 1,
    sets = [<0, bindings = [
        <0, storage_buffer, ReadOnly>,
        <1, storage_buffer, ReadOnly>,
        <2, storage_buffer, ReadOnly>,
        <3, storage_buffer>
    ]>]>

#bindings = [
    #hal.interface.binding<0, 0>,
    #hal.interface.binding<0, 1>,
    #hal.interface.binding<0, 2>,
    #hal.interface.binding<0, 3>
]
module {
  flow.executable private @forward_dispatch_fff_0 {
    flow.executable.export public @forward_dispatch_fff_0 workgroups(%arg0: !hal.device, %arg1: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %arg1, %c1, %c1 : index, index, index
    } attributes {hal.interface.layout = #pipeline_layout,
                  translation_info = #iree_codegen.translation_info<SPIRVSubgroupReduce>,
                  workgroup_size = [64 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @forward_dispatch_fff_0(%binding0: !stream.binding,
                                        %binding1: !stream.binding,
                                        %binding2: !stream.binding,
                                        %binding3: !stream.binding,
                                        %batch_size: index) {
        %c32_i64 = arith.constant 32 : i64
        %c0_f32 = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index

        %d1 = arith.constant 12 : index

        %in_binding = stream.binding.subspan %binding0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:!full_in_t>{%batch_size}
        %w1_binding = stream.binding.subspan %binding1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:!w1_t>
        %w2_binding = stream.binding.subspan %binding2[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:!w2_t>
        %dest_binding = stream.binding.subspan %binding3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:!full_out_t>{%batch_size}
        %workgroup_id_x = flow.dispatch.workgroup.id[0] : index

        %in = flow.dispatch.tensor.load %in_binding, offsets = [%workgroup_id_x, 0], sizes = [1, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:!full_in_t>{%batch_size} -> !in_t
        %w1 = flow.dispatch.tensor.load %w1_binding, offsets = [0, 0], sizes = [4095, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:!w1_t> -> !w1_t
        %w2 = flow.dispatch.tensor.load %w2_binding, offsets = [0, 0], sizes = [4095, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:!w2_t> -> !w2_t

        %empty = tensor.empty() : !out_t
        %init = linalg.fill {
                    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [64]]>
                } ins(%c0_f32: f32) outs(%empty: !out_t) -> !out_t

        %s:2 = scf.for %i = %c0 to %d1 step %c1 iter_args(%curr = %c0, %acc = %init) -> (index, !out_t) {
          %w1_row = tensor.extract_slice %w1[%curr, 0][1, 768][1, 1] : !w1_t to !in_t

          %score_empty = tensor.empty() : !score_t
          %score_init = linalg.fill ins(%c0_f32: f32) outs(%score_empty: !score_t) -> !score_t
          %score_tensor = linalg.generic {
                indexing_maps = [
                    affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> ()>
                ], iterator_types = ["reduction"]}
            ins(%in, %w1_row: !in_t, !in_t) outs(%score_init: !score_t)
            attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [64]]>} {
              ^bb0(%l: f32, %r: f32, %out: f32):
                %mul = arith.mulf %l, %r : f32
                %add = arith.addf %mul, %out: f32
                linalg.yield %add : f32
          } -> !score_t
          %score = tensor.extract %score_tensor[] : !score_t

          %choice = arith.cmpf oge, %score, %c0_f32 : f32
          %choice_i64 = arith.extui %choice : i1 to i64
          %choice_index = arith.index_cast %choice_i64 : i64 to index
          %branch = arith.muli %curr, %c2 : index
          %inc = arith.addi %branch, %c1: index
          %next = arith.addi %inc, %choice_index: index

          %mean = arith.constant 0.0 : f32
          %sigma = arith.constant 1.0 : f32
          %c_half_f32 = arith.constant 0.5 : f32
          %c2_f32 = arith.constant 2.0 : f32
          %sub = arith.subf %score, %mean : f32
          %sq2 = math.sqrt %c2_f32 : f32
          %div = arith.divf %sub, %sq2 : f32
          %erf = math.erf %div : f32
          %inc_f32 = arith.addf %erf, %sigma : f32
          %gelu = arith.mulf %inc_f32, %c_half_f32 : f32

          %w2_row = tensor.extract_slice %w2[%curr, 0][1, 768][1, 1] : !w2_t to !out_t
          %new_out = linalg.generic {
                indexing_maps = [
                    affine_map<(d0) -> (d0)>,
                    affine_map<(d0) -> (d0)>
                ], iterator_types = ["parallel"]}
            ins(%w2_row: !out_t) outs(%acc: !out_t)
            attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [64]]>} {
              ^bb0(%w: f32, %out: f32):
                %mul = arith.mulf %w, %gelu : f32
                %add = arith.addf %mul, %out: f32
                linalg.yield %add : f32
          } -> !out_t
          scf.yield %next, %new_out : index, !out_t
        }
        flow.dispatch.tensor.store %s#1, %dest_binding, offsets = [%workgroup_id_x, 0], sizes = [1, 768], strides = [1, 1] : !out_t -> !flow.dispatch.tensor<writeonly:!full_out_t>{%batch_size}
        return
      }
    }
  }
  func.func @forward(%in: !full_in_t, %w1: !w1_t, %w2: !w2_t) -> !full_out_t {
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %in, %c0 : !full_in_t
    %batch_i64 = arith.index_cast %batch : index to i64
    %0 = flow.dispatch @forward_dispatch_fff_0::@forward_dispatch_fff_0[%batch](%batch_i64, %in, %w1, %w2) {hal.interface.bindings = #bindings} : (i64, !full_in_t{%batch}, !w1_t, !w2_t) -> !full_out_t{%batch}
    return %0 : !full_out_t
  }
}

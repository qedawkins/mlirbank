// in_width = 256
// depth = 10
// out_width = 128
// n_nodes = 2 ** (depth + 1) - 1 = 2047
// batch_size = 64

!w1_t = tensor<2047x256xf32>
!w2_t = tensor<2047x128xf32>
!in_t = tensor<64x256xf32>
!select_t = tensor<64x11x128xf32>
!score_t = tensor<64xf32>
!nodes_t = tensor<64xindex>
!gather_nodes_t = tensor<64x1xindex>
!logit_t = tensor<64x11xf32>
!choice_t = tensor<64x11xindex>
!gather_choice_t = tensor<64x11x1xindex>
!out_t = tensor<64x128xf32>

func.func @fff(%in: !in_t, %w1: !w1_t, %w2: !w2_t) -> !out_t {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_f32 = arith.constant 0.0 : f32
  %d1 = arith.constant 11 : index

  %curr_empty = tensor.empty() : !nodes_t
  %curr_init = linalg.fill ins(%c0: index) outs(%curr_empty: !nodes_t) -> !nodes_t

  %rows_empty = tensor.empty() : !choice_t
  %select_init = linalg.fill ins(%c0: index) outs(%rows_empty: !choice_t) -> !choice_t

  %logits_empty = tensor.empty() : !logit_t
  %logits_init = linalg.fill ins(%c0_f32: f32) outs(%logits_empty: !logit_t) -> !logit_t

  %s:3 = scf.for %i = %c0 to %d1 step %c1
             iter_args(%curr = %curr_init, %select = %select_init, %logits = %logits_init)
             -> (!nodes_t, !choice_t, !logit_t) {
    %new_select = tensor.insert_slice %curr into %select[0, %i][64, 1][1, 1] : !nodes_t into !choice_t
    %expanded_select = tensor.expand_shape %curr [[0, 1]] : !nodes_t into !gather_nodes_t
    %select_weights = tensor.gather %w1[%expanded_select] gather_dims([0]) unique : (!w1_t, !gather_nodes_t) -> !in_t

    %score_empty = tensor.empty() : !score_t
    %score_init = linalg.fill ins(%c0_f32: f32) outs(%score_empty: !score_t) -> !score_t
    %scores = linalg.generic {
          indexing_maps = [
              affine_map<(d0, d1) -> (d0, d1)>,
              affine_map<(d0, d1) -> (d0, d1)>,
              affine_map<(d0, d1) -> (d0)>
          ], iterator_types = ["parallel", "reduction"]}
      ins(%in, %select_weights: !in_t, !in_t) outs(%score_init: !score_t) {
        ^bb0(%l: f32, %r: f32, %out: f32):
          %mul = arith.mulf %l, %r : f32
          %add = arith.addf %mul, %out: f32
          linalg.yield %add : f32
    } -> !score_t

    %new_logits = tensor.insert_slice %scores into %logits[0, %i][64, 1][1, 1] : !score_t into !logit_t

    %new_curr_empty = tensor.empty() : !nodes_t
    %new_current = linalg.generic {
          indexing_maps = [
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>,
              affine_map<(d0) -> (d0)>
          ], iterator_types = ["parallel"]}
      ins(%curr, %scores: !nodes_t, !score_t) outs(%new_curr_empty: !nodes_t) {
        ^bb0(%c: index, %score: f32, %out: index):
          %choice = arith.cmpf oge, %score, %c0_f32 : f32
          %choice_index = arith.index_cast %choice : i1 to index
          %branch = arith.muli %c, %c2 : index
          %inc = arith.addi %branch, %c1: index
          %next = arith.addi %inc, %choice_index: index
          linalg.yield %next : index
    } -> !nodes_t

    scf.yield %new_current, %new_select, %new_logits : !nodes_t, !choice_t, !logit_t
  }

  %expanded_select_2 = tensor.expand_shape %s#1 [[0], [1, 2]] : !choice_t into !gather_choice_t
  %selected_w2 = tensor.gather %w2[%expanded_select_2] gather_dims([0]) unique : (!w2_t, !gather_choice_t) -> !select_t

  %mean = arith.constant 0.0 : f32
  %sigma = arith.constant 1.0 : f32
  %c_half_f32 = arith.constant 0.5 : f32
  %c2_f32 = arith.constant 2.0 : f32
  %activations = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]}
    ins(%s#2: !logit_t) outs(%logits_empty: !logit_t) {
      ^bb0(%logit: f32, %out: f32):
        %sub = arith.subf %logit, %mean : f32
        %sq2 = math.sqrt %c2_f32 : f32
        %div = arith.divf %sub, %sq2 : f32
        %erf = math.erf %div : f32
        %inc = arith.addf %erf, %sigma : f32
        %gelu = arith.mulf %inc, %c_half_f32 : f32
        linalg.yield %gelu : f32
  } -> !logit_t

  %empty = tensor.empty() : !out_t
  %init = linalg.fill ins(%c0_f32: f32) outs(%empty: !out_t) -> !out_t
  %result = linalg.batch_vecmat
              ins(%activations, %selected_w2: !logit_t, !select_t)
              outs(%init: !out_t) -> !out_t
  return %result : !out_t
}

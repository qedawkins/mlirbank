// in_width = 256
// depth = 10
// out_width = 128
// n_nodes = 2 ** (depth + 1) - 1 = 2047
// batch_size = 64

!w1_t = tensor<768x3072xf32>
!w2_t = tensor<3072x768xf32>
!in_t = tensor<16384x768xf32>
!inter_t = tensor<16384x3072xf32>
!out_t = tensor<16384x768xf32>

func.func @forward(%in: !in_t, %w1: !w1_t, %w2: !w2_t) -> !out_t {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_f32 = arith.constant 0.0 : f32
  %d1 = arith.constant 11 : index

  %inter_empty = tensor.empty() : !inter_t
  %inter_init = linalg.fill ins(%c0_f32: f32) outs(%inter_empty: !inter_t) -> !inter_t
  %m1 = linalg.matmul ins(%in, %w1: !in_t, !w1_t) outs(%inter_init: !inter_t) -> !inter_t

  %mean = arith.constant 0.0 : f32
  %sigma = arith.constant 1.0 : f32
  %c_half_f32 = arith.constant 0.5 : f32
  %c2_f32 = arith.constant 2.0 : f32
  %activations = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]}
    ins(%m1: !inter_t) outs(%inter_empty: !inter_t) {
      ^bb0(%logit: f32, %out: f32):
        %sub = arith.subf %logit, %mean : f32
        %sq2 = math.sqrt %c2_f32 : f32
        %div = arith.divf %sub, %sq2 : f32
        %erf = math.erf %div : f32
        %inc = arith.addf %erf, %sigma : f32
        %gelu = arith.mulf %inc, %c_half_f32 : f32
        linalg.yield %gelu : f32
  } -> !inter_t

  %empty = tensor.empty() : !out_t
  %init = linalg.fill ins(%c0_f32: f32) outs(%empty: !out_t) -> !out_t
  %result = linalg.matmul
              ins(%activations, %w2: !inter_t, !w2_t)
              outs(%init: !out_t) -> !out_t
  return %result : !out_t
}

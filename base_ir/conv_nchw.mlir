#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "Conv2d"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x4x32x32xf32> {
    %cst = arith.constant dense<[[[[-0.194130942], [-0.002053231], [-0.231181473]], [[-3.029260e-02], [-0.00984880328], [0.17870906]], [[-0.056172967], [-4.834640e-02], [-0.269018412]]], [[[-0.0401979685], [0.310443789], [-0.0480653942]], [[-0.0493329465], [-0.118616432], [-0.215419739]], [[0.0980359315], [-8.195710e-02], [-0.229281545]]], [[[0.128511041], [-0.159413904], [-0.234531641]], [[-0.0471769273], [0.241814166], [-0.0965808629]], [[-0.311713904], [0.293067485], [1.79409981E-4]]], [[[-0.245140284], [-0.311403513], [-0.218301624]], [[0.139625669], [9.060490e-02], [0.118851066]], [[-0.224677771], [-0.112303421], [-0.0881195962]]]]> : tensor<4x3x3x1xf32>
    %cst_0 = arith.constant dense<[-0.311590075, 0.249434024, 0.314933091, -5.043140e-02]> : tensor<4xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %padded = tensor.pad %arg0 low[0, 0, 1, 0] high[0, 0, 1, 0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_1 : f32
    } : tensor<1x3x32x32xf32> to tensor<1x3x34x32xf32>
    %0 = tensor.empty() : tensor<1x4x32x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<4xf32>) outs(%0 : tensor<1x4x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x4x32x32xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %cst : tensor<1x3x34x32xf32>, tensor<4x3x3x1xf32>) outs(%1 : tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    return %2 : tensor<1x4x32x32xf32>
  }
}

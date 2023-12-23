//#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx90a"}>
//#device_target_rocm = #hal.device.target<"rocm", {executable_targets = [#executable_target_rocm_hsaco_fb], legacy_sync}>
//module attributes {hal.device.targets = [#device_target_rocm], torch.debug_module_name = "_lambda"} {
//  func.func @forward(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
//    %c128 = arith.constant 128 : index
//    %c4 = arith.constant 4 : index
//    %c1 = arith.constant 1 : index
//    %cst = arith.constant 0.000000e+00 : f16
//    %c0 = arith.constant 0 : index
//    %0 = flow.tensor.splat %cst : tensor<2x4x128x128xf16>
//    %1 = flow.tensor.slice %0[%c0, %c0, %c0, %c0 for %c1, %c4, %c128, %c128] : tensor<2x4x128x128xf16> -> tensor<1x4x128x128xf16>
//    %2 = hal.tensor.export %1 "output 0" : tensor<1x4x128x128xf16> -> !hal.buffer_view
//    return %2 : !hal.buffer_view
//  }
//}

//module {
//  func.func @forward(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
//    %c128 = arith.constant 128 : index
//    %c4 = arith.constant 4 : index
//    %c1 = arith.constant 1 : index
//    %cst = arith.constant 0.000000e+00 : f16
//    %c0 = arith.constant 0 : index
//    %0 = flow.tensor.splat %cst : tensor<2x4x128x128xf16>
//    %1 = flow.tensor.slice %0[%c0, %c0, %c0, %c0 for %c1, %c4, %c128, %c128] : tensor<2x4x128x128xf16> -> tensor<1x4x128x128xf16>
//    %2 = hal.tensor.export %1 "output 0" : tensor<1x4x128x128xf16> -> !hal.buffer_view
//    return %2 : !hal.buffer_view
//  }
//}

module {
  func.func @forward(%arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %0 = flow.tensor.splat %cst : tensor<2x4x128x128xf16>
    %1 = flow.tensor.slice %0[%c0, %c0, %c0, %c0 for %c1, %c4, %c128, %c128] : tensor<2x4x128x128xf16> -> tensor<1x4x128x128xf16>
    %2 = hal.tensor.export %1 "output 0" : tensor<1x4x128x128xf16> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}

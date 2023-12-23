hal.executable public @second_vicuna_forward_dispatch_30 {
  hal.executable.variant public @embedded_elf_x86_64 target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf", ukernels = false}>) {
    hal.executable.export public @second_vicuna_forward_dispatch_30_generic_4096x86x128_f16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 5, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @second_vicuna_forward_dispatch_30_generic_4096x86x128_f16() {
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %3 = hal.interface.constant.load[3] : i32
        %4 = hal.interface.constant.load[4] : i32
        %5 = arith.index_castui %0 : i32 to index
        %6 = arith.index_castui %1 : i32 to index
        %7 = arith.index_castui %2 : i32 to index
        %8 = arith.index_castui %3 : i32 to index
        %9 = arith.index_castui %4 : i32 to index
        %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
        %11 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>>
        %12 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%7) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>>
        %13 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%8) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<86x128xf16>>
        %14 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%9) : !flow.dispatch.tensor<readwrite:tensor<4096xf16>>
        %15 = flow.dispatch.tensor.load %10, offsets = [0, 0, 0], sizes = [4096, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<4096x86x128xi4>
        %16 = flow.dispatch.tensor.load %11, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>> -> tensor<4096x86xf16>
        %17 = flow.dispatch.tensor.load %12, offsets = [0, 0], sizes = [4096, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf16>> -> tensor<4096x86xf16>
        %18 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf16>> -> tensor<86x128xf16>
        %19 = flow.dispatch.tensor.load %14, offsets = [0], sizes = [4096], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4096xf16>> -> tensor<4096xf16>
        %20 = tensor.empty() : tensor<4096x86x128xf16>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16, %17 : tensor<4096x86x128xi4>, tensor<4096x86xf16>, tensor<4096x86xf16>) outs(%20 : tensor<4096x86x128xf16>) {
        ^bb0(%in: i4, %in_0: f16, %in_1: f16, %out: f16):
          %23 = arith.extui %in : i4 to i32
          %24 = arith.uitofp %23 : i32 to f16
          %25 = arith.subf %24, %in_1 : f16
          %26 = arith.mulf %25, %in_0 : f16
          linalg.yield %26 : f16
        } -> tensor<4096x86x128xf16>
        %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"]} ins(%18, %21 : tensor<86x128xf16>, tensor<4096x86x128xf16>) outs(%19 : tensor<4096xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %23 = arith.mulf %in, %in_0 : f16
          %24 = arith.addf %23, %out : f16
          linalg.yield %24 : f16
        } -> tensor<4096xf16>
        flow.dispatch.tensor.store %22, %14, offsets = [0], sizes = [4096], strides = [1] : tensor<4096xf16> -> !flow.dispatch.tensor<readwrite:tensor<4096xf16>>
        return
      }
    }
  }
}

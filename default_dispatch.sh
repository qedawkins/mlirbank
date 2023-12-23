#~/iree-build/tools/iree-compile base_ir/glsl_f16argmax.mlir \
#~/iree-build/tools/iree-compile base_ir/torch_gemm.mlir \
#~/iree-build/tools/iree-compile base_ir/matmul.mlir \
#~/iree-build/tools/iree-compile base_ir/quant_matvec.mlir \
#~/iree-build/tools/iree-compile base_ir/quant_double_matvec.mlir \
#~/iree-build/tools/iree-compile base_ir/glsl_f16argmax.mlir \
#~/iree-build/tools/iree-compile base_ir/glsl_extern_f16_argmax.mlir \
#~/iree-build/tools/iree-compile base_ir/flow_extern_fallback.mlir \
#~/iree-build/tools/iree-compile base_ir/glsl_extern_f16_argmax.mlir \
#~/iree-build/tools/iree-compile base_ir/quant_matvec.mlir \
#~/iree-build/tools/iree-compile base_ir/example.mlir \
#~/iree-build/tools/iree-compile base_ir/conv_nchw.mlir \
#~/iree-build/tools/iree-compile base_ir/rocm_example.mlir \
#~/iree-build/tools/iree-compile base_ir/argmax.mlir \
#~/iree-build/tools/iree-compile base_ir/dynamic_reduction.mlir \
#~/iree-build/tools/iree-compile base_ir/bert_FF.mlir \
#~/iree-build/tools/iree-compile base_ir/hoist_test.mlir \
#~/iree-build/tools/iree-compile base_ir/collapsible.mlir \
#~/iree-build/tools/iree-compile base_ir/elementwise.mlir \
#~/iree-build/tools/iree-compile base_ir/spirv_FFF.mlir \
#~/iree-build/tools/iree-compile base_ir/stream_FFF.mlir \
#~/iree-build/tools/iree-compile base_ir/dynamic_softmax.mlir \
~/iree-build/tools/iree-compile base_ir/glsl_extern_f16_argmax.mlir \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    --iree-vulkan-target-env="#vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, #vk.caps<maxComputeSharedMemorySize = 16384, maxComputeWorkGroupInvocations = 1024, maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>, subgroupFeatures = 0 : i32, subgroupSize = 4 >>" \
    --iree-vulkan-target-triple=rdna1-unknown-linux \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-rocm-target-chip=gfx1100 \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-stream-resource-max-allocation-size=4294967295 \
    --iree-rocm-link-bc=true \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-hal-target-backends=vulkan \
    --mlir-disable-threading \
    --iree-codegen-llvmgpu-enable-transform-dialect-jit=true \
    --iree-hal-benchmark-dispatch-repeat-count=100 \
    --mlir-print-ir-before-all \
    --mlir-print-ir-after-all \
    -o /tmp/dispatch.vmfb
    #--iree-hal-instrument-dispatches=16mib \
    #--iree-llvmcpu-instrument-memory-accesses=true \
    #--iree-flow-trace-dispatch-tensors \
    #--debug \
    #--debug-only=transform-dialect-print-top-level-after-all \
    #--debug-only=transform-dialect \
    #--debug-only=iree-codegen-vector-reduction-to-gpu \
    #--iree-scheduling-optimize-bindings=false \
    #--compile-from=flow \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-vulkan-target-env="#vk.target_env<v1.1, r(120), [VK_KHR_spirv_1_4, VK_KHR_storage_buffer_storage_class], AMD:DiscreteGPU, #vk.caps<maxComputeSharedMemorySize = 16384, maxComputeWorkGroupInvocations = 1024, maxComputeWorkGroupSize = dense<[128, 8, 4]>: vector<3xi32>, subgroupFeatures = 63 : i32, subgroupSize = 4 >>" \
    #--iree-vulkan-target-triple=rdna1-unknown-linux \
    #--iree-hal-benchmark-dispatch-repeat-count=100 \
    #--mlir-print-ir-before-all \
    #--mlir-print-ir-after-all \
    #--iree-hal-dump-executable-sources-to=sources \
    #--iree-hal-dump-executable-binaries-to=binaries \

~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
  --function=forward \
  --input=1024x1024xf32 \
  --input=1024x1024xf32 \
  --batch_size=100 \
  --module=/tmp/dispatch.vmfb

#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --function=forward \
#  --input=16384x768xf32 \
#  --input=4095x768xf32 \
#  --input=4095x768xf32 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb

#~/iree-build/tools/iree-benchmark-module --device=rocm --function=forward \
#  --input=32x256xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb

#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x256xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#
#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x15xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#
#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x1xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#
#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x127xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#
#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x128xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#
#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=forward \
#  --input=32x850xf16=1.0 \
#  --batch_size=100 \
#  --module=/tmp/dispatch.vmfb
#  #--input=1x32000xf16 \
#  #--input=2x4096x32xf16 \
#  #--input=2x4096x32xf16 \
#  #--input=2x32x128xf16 \
#  #--input=32x300xf16 \
#  #--input=1x32000xf16 \
#  #--batch_size=100 \
#  #--benchmark_repetitions=10 \
#  #--input=4096x32xf16 \
#  #--input=4096x32xf16 \
#  #--input=32x128xf16 \

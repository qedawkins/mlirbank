#~/iree-build/tools/iree-compile models/llama2_70b_folded.mlir \
#~/iree-build/tools/iree-compile models/llama2_7b_argmax_folded.mlir \
#~/iree-build/tools/iree-compile models/llama2_7b_folded.mlir \
#~/iree-build/tools/iree-compile models/llama2_13b_int4_stripped.mlir \
~/iree-build/tools/iree-compile models/llama2_7b_folded.mlir \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-rocm-target-chip=gfx1100 \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-stream-resource-max-allocation-size=4294967295 \
    --iree-hal-target-backends=rocm \
    --iree-hal-dump-executable-sources-to=rocm_sources \
    -o /tmp/dispatch.vmfb
    #--iree-hal-dump-executable-sources-to=sources \
    #--iree-hal-dump-executable-binaries-to=binaries \
    #--mlir-print-ir-after-all \
    #--compile-to=flow \
    #-o /tmp/dispatch.mlir

#~/iree-build/tools/iree-compile models/llama2_7b_folded.mlir \
#    --iree-vulkan-target-triple=rdna3-unknown-linux \
#    --iree-vulkan-target-triple=ampere-unknown-linux \
#    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
#    --iree-hal-cuda-llvm-target-arch=sm_80 \
#    --iree-rocm-target-chip=gfx1100 \
#    --iree-stream-resource-index-bits=64 \
#    --iree-vm-target-index-bits=64 \
#    --iree-input-type=none \
#    --iree-stream-resource-max-allocation-size=4294967295 \
#    --iree-hal-target-backends=vulkan \
#    -o vmfbs/multiple.vmfb

#~/iree-build/tools/iree-benchmark-module --device=vulkan --function=first_vicuna_forward \
#  --input=1x16xi64=[[1 1128 1784 11192 947 263 17205 505 29973 673 297 3109 1135 29871 29953 3838]]
TRACY_NO_EXIT=1 ~/iree-tracy-build/tools/iree-benchmark-module --device=rocm --function=second_vicuna_forward \
  --module=/tmp/dispatch.vmfb \
  --input=1x1xi64 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16 \
  --input=1x32x100x128xf16

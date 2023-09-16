iree-compile base_ir/elementwise.mlir \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-rocm-target-chip=gfx1100 \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-hal-target-backends=vulkan-spirv \
    --iree-hal-benchmark-dispatch-repeat-count=100 \
    -o /tmp/dispatch.vmfb

iree-benchmark-module --device=vulkan --function=forward \
  --input=1024x1024xf32 \
  --input=1024x1024xf32 \
  --batch_size=100 \
  --module=/tmp/dispatch.vmfb

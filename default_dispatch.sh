~/iree-build/tools/iree-compile base_ir/quant_matvec.mlir \
    --iree-vulkan-target-triple=rdna3-unknown-linux \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-rocm-target-chip=gfx1100 \
    --iree-stream-resource-index-bits=64 \
    --iree-vm-target-index-bits=64 \
    --iree-input-type=none \
    --iree-stream-resource-max-allocation-size=4294967295 \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
    --iree-hal-target-backends=rocm \
    --iree-scheduling-optimize-bindings=false \
    --iree-hal-dump-executable-sources-to=sources \
    --iree-hal-dump-executable-binaries-to=binaries \
    --iree-hal-benchmark-dispatch-repeat-count=100 \
    -o /tmp/dispatch.vmfb

TRACY_NO_EXIT=1 ~/iree-build/tools/iree-benchmark-module --device=rocm --function=forward \
  --module=/tmp/dispatch.vmfb \
  --input=4096x32xf16 \
  --input=4096x32xf16 \
  --input=32x128xf16 \
  --benchmark_time_unit=ns \
  --batch_size=100

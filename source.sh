for NUM in {3..3}
do
  FNAME_BASE="$(find sources -name module_second_vicuna_forward_dispatch_${NUM}.mlir)"
  if (($(echo $FNAME_BASE | grep -c . ) == 1)); then
    echo "Testing $FNAME_BASE"
    ~/iree-build/tools/iree-compile \
        --iree-input-type=none \
        --iree-hal-target-backends=llvm-cpu \
        --iree-vulkan-target-triple=rdna3-unknown-linux \
        --iree-llvmcpu-target-triple=x86_64-unknown-linux \
        --iree-rocm-target-chip=gfx1100 \
        --iree-stream-resource-index-bits=64 \
        --iree-vm-target-index-bits=64 \
        --iree-rocm-link-bc=true \
        --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode \
        --iree-hal-target-backends=rocm \
        --compile-from=flow \
        --debug-only=kernel-dispatch \
        --mlir-print-ir-after-all \
        $FNAME_BASE \
        -o /tmp/module.vmfb
  else
    echo "Could not find single source for ${NUM}"
  fi
done

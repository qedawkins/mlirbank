#~/iree-build/tools/iree-compile --output-format=vm-bytecode --mlir-print-op-on-diagnostic=false --iree-hal-target-backends=llvm-cpu --iree-input-type=none --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu --iree-llvmcpu-target-cpu=cascadelake --iree-opt-data-tiling --iree-llvmcpu-enable-microkernels models/falcon7b_gptq_linalg_zeroed_weights_1698708010.mlirbc -o /tmp/dispatch.vmfb
~/iree-build/tools/iree-compile \
    --output-format=vm-bytecode \
    --mlir-print-op-on-diagnostic=false \
    --iree-hal-target-backends=llvm-cpu \
    --iree-input-type=none \
    --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
    --iree-llvmcpu-target-cpu=cascadelake \
    --iree-opt-data-tiling \
    --iree-llvmcpu-enable-microkernels \
    models/falcon7b_gptq_linalg_zeroed_weights_1698708010.mlirbc \
    -o /tmp/dispatch.vmfb
    #--mlir-print-ir-after=iree-util-hoist-into-globals \

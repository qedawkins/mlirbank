#~/iree-build/tools/iree-compile hello_detokenizer.mlir \
#~/iree-build/tools/iree-compile printer_detokenizer.mlir \
~/iree-build/tools/iree-compile global_detokenizer.mlir \
    --iree-hal-target-backends=vmvx \
    --mlir-print-ir-after-all \
    -o /tmp/dispatch.vmfb

~/iree-build/tools/iree-run-module --device=local-task --function=forward \
  --function=detokenize \
  --input=4xi32=[1,0,2,2] \
  --module=/tmp/dispatch.vmfb

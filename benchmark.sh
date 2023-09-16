for NUM in {1..10}
do
  FNAME_BASE="$(find benchmarks -name module_forward_dispatch_${NUM}_*)"
  if (($(echo $FNAME_BASE | grep -c . ) == 1)); then
    echo "Testing $FNAME_BASE"
    iree-compile \
        --iree-input-type=none \
        --iree-hal-target-backends=vulkan \
        --iree-vulkan-target-triple=rdna3-unknown-linux \
        --iree-stream-resource-index-bits=64 \
        --iree-vm-target-index-bits=64 \
        --iree-hal-benchmark-dispatch-repeat-count=100 \
        $FNAME_BASE \
        -o /tmp/module.vmfb

    FUNCTION="$(cat $FNAME_BASE | grep iree.benchmark | grep -o -o '[^ @]*_dispatch[^ (]*')"
    echo "Compiled, trying with function $FUNCTION"
    
    iree-benchmark-module \
        --module=/tmp/module.vmfb \
        --function=$FUNCTION \
        --input=1 \
        --batch_size=100 \
        --device=vulkan
  else
    echo "Could not find single benchmark for ${NUM}"
  fi
done

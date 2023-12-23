import torch
import numpy as np

#example_input = torch.randn([1, 32000])
example_input = torch.randn([1, 32000]).to(torch.half)
print(example_input)

# Run through IREE vulkan
import iree.runtime as ireert
from iree.runtime import get_driver, get_device

with open("/tmp/dispatch.vmfb", "rb") as f:
    flatbuffer = f.read()

runtime_device = "vulkan"

config = ireert.Config(driver_name=runtime_device)
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, flatbuffer)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
forward = ctx.modules.module["forward"]
res = forward(example_input.numpy())
iree_output = torch.from_numpy(res.to_host())

print(iree_output)
print(torch.argmax(example_input))

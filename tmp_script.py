import torch

gpu_props = torch.cuda.get_device_properties(0)
print(dir(gpu_props))

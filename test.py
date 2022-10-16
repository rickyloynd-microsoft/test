import time
import math
import torch

start_time = time.time()

print('{:4.1f} s  PyTorch version      {}'.format(time.time() - start_time, torch.version.__version__))
print('{:4.1f} s  CUDA version         {}'.format(time.time() - start_time, torch.version.cuda))

try_to_use_cuda = 1

cuda_is_available = torch.cuda.is_available()
print('{:4.1f} s  CUDA available       {}'.format(time.time() - start_time, cuda_is_available))

# Choose the device.
device_str = 'cpu'
if try_to_use_cuda:
    if cuda_is_available:
        device_str = 'cuda'
device = torch.device(device_str)
print('{:4.1f} s  device               {}'.format(time.time() - start_time, device))

if device_str == 'cuda':
    torch.cuda.init()
    props = torch.cuda.get_device_properties(device)
    print('{:4.1f} s  CUDA GPU             {}'.format(time.time() - start_time, props.name))
    print('{:4.1f} s  CUDA GPU RAM         {:3.1f} GB'.format(time.time() - start_time, props.total_memory / math.pow(1024., 3)))
    print('{:4.1f} s  CUDA GPU processors  {}'.format(time.time() - start_time, props.multi_processor_count))
    print('{:4.1f} s  CUDA device count    {}'.format(time.time() - start_time, torch.cuda.device_count()))
    print('{:4.1f} s  CUDA current device  {}'.format(time.time() - start_time, torch.cuda.current_device()))
    torch.manual_seed(1)
    print('{:4.1f} s  {}'.format(time.time() - start_time, torch.rand(1).cuda()))

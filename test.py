import torch

print('PyTorch version      {}'.format(torch.version.__version__))
print('CUDA version         {}'.format(torch.version.cuda))

try_to_use_cuda = 1

cuda_is_available = torch.cuda.is_available()
print('CUDA available       {}'.format(cuda_is_available))

# Choose the device.
device_str = 'cpu'
if try_to_use_cuda:
    if cuda_is_available:
        device_str = 'cuda'
device = torch.device(device_str)
print('device               {}'.format(device))

if device_str == 'cuda':
    torch.cuda.init()
    props = torch.cuda.get_device_properties(device)
    print('CUDA GPU             {}'.format(props.name))
    print('CUDA GPU RAM         {}'.format('{:3.1f} GB'.format(props.total_memory / 1024. / 1024. / 1024.)))
    print('CUDA GPU processors  {}'.format(props.multi_processor_count))
    print('CUDA device count    {}'.format(torch.cuda.device_count()))
    print('CUDA current device  {}'.format(torch.cuda.current_device()))
    torch.manual_seed(1)
    print(torch.rand(1).cuda())

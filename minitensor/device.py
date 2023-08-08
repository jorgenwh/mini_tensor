from .tensor import Tensor

def to_cpu(tensor):
    return tensor.to_cpu()

def to_cuda(tensor):
    return tensor.to_cuda()

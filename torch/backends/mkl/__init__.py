import torch

def is_available():
    r"""Returns whether PyTorch is built with MKL support."""
    return torch._C.has_mkl

class verbose(object):
    def __init__(self, enable=0):
        self.enable = enable
        self.status = -1

    def __enter__(self):
        if self.enable == 0:
            return
        self.status = torch._C._verbose.mkl_set_verbose(self.enable)
        if self.status == -1:
            print('[Warning] Failed to enable MKL verbose.')
            return
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.status == -1:
            return
        else:
            torch._C._verbose.mkl_set_verbose(0)
            return False

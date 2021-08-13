import torch

def is_available():
    r"""Returns whether PyTorch is built with MKL support."""
    return torch._C.has_mkl

VERBOSE_OFF = 0
VERBOSE_ON  = 1
class verbose(object):
    def __init__(self, enable):
        self.enable = enable

    def __enter__(self):
        if self.enable == MKL_VERBOSE_OFF:
            return
        st = torch._C._verbose.mkl_set_verbose(self.enable)
        assert st, "Failed to set MKL into verbose mode. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkl_set_verbose(MKL_VERBOSE_OFF)
        return False

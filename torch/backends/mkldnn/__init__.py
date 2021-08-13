import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn

class verbose(object):
    def __init__(self, level=0):
        self.level = level
        self.status = 0

    def __enter__(self):
        if self.level == 0:
            return
        self.status = torch._C._verbose.mkldnn_set_verbose(self.level)
        if self.status != 1:
            print('[Warning] Failed to enable MKLDNN verbose.')
            return
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.status != 1:
            return
        else:
            torch._C._verbose.mkldnn_set_verbose(0)
            return False

def set_flags(_enabled):
    orig_flags = (torch._C._get_mkldnn_enabled(),)
    torch._C._set_mkldnn_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])

class MkldnnModule(PropModule):
    def __init__(self, m, name):
        super(MkldnnModule, self).__init__(m, name)

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)

# Cool stuff from torch/backends/cudnn/__init__.py and
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)

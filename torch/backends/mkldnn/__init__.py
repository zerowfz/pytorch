import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn

VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2
class verbose(object):
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == MKLDNN_VERBOSE_OFF:
            return
        st = torch._C._verbose.mkldnn_set_verbose(self.level)
        assert st, "Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkldnn_set_verbose(MKLDNN_VERBOSE_OFF)
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

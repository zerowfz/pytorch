import argparse
import torch

def run_model(level):
    tensor1 = torch.randn(10, 3, 4)
    tensor2 = torch.randn(10, 4, 5)
    with torch.backends.mkl.verbose(level):
        torch.matmul(tensor1, tensor2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    run_model(args.verbose_level)

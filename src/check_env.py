"""Print installed package versions and CUDA availability."""
import torch
import numpy
import matplotlib
import sklearn

print("python packages:")
print(f"  torch:      {torch.__version__}")
print(f"  torch.cuda: {torch.cuda.is_available()} (available)")
print(f"  numpy:      {numpy.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
print(f"  sklearn:    {sklearn.__version__}")
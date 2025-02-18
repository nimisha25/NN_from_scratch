import torch.nn as nn
m = nn.BatchNorm1d(15)    # creating an instance of the Batch Normalization module for 1D inputs with 15 features

# learnable affine parameters (a and b)

print(m.weight , m.bias)
print(m.running_mean , m.running_var)
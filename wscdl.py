"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *

opts = OPT()
X, Y = load_data()
D, D0, S, S0, W = init(opts)

for i in range(opts.maxiter):
    D = updateD([D, D0, S, S0], X, Y, opts)
    D0 = updateD0([D, D0, S, S0], X, Y, opts)
    S = updateS([D, D0, S, S0, W], X, Y, opts)
    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    W = updateW([S, W], Y, opts)

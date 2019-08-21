"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *

opts = OPT()
X, Y = load_data(opts)
D, D0, S, S0, W = init(X, Y, opts)

for i in range(opts.maxiter):
    t = time.time()
    D = updateD([D, D0, S, S0], X, Y, opts)
    print('pass D, time is ', time.time() -t ); t = time.time()
    D0 = updateD0([D, D0, S, S0], X, Y, opts)
    print('pass D0, time is ', time.time() -t); t = time.time()
    S = updateS([D, D0, S, S0, W], X, Y, opts)
    print('pass S, time is ', time.time() -t); t = time.time()
    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    print('pass S0, time is ', time.time() -t); t = time.time()
    W = updateW([S, W], Y, opts)
    print('pass W, time is ', time.time() -t)

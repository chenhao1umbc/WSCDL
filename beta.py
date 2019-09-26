"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
opts = OPT()
opts.snr = 100
opts.lamb = 3 # for sparsity penalty
opts.eta = 10  # for label penalty
opts.mu = 0  # for low rank penalty
opts.show_details = False

# training section
X, Y, ft = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
if opts.show_details:  # default True
    D, D0, S, S0, W, loss = train_details(D, D0, S, S0, W, X, Y, opts)
else:
    D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
X_test, Y_test, _ = load_toy(opts)
_, _, S_t, S0_t, _ = init(X_test, opts)
if opts.show_details:  # default True
    acc, y_hat = test_details(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
else:
    acc, y_hat = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('The test data accuracy is : ', acc)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft, loss, opts)
print('done')
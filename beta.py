"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
opts = OPT()
opts.lamb = 10  # for sparsity penalty
opts.eta = 100  # for label penalty
opts.mu = 1  # for low rank penalty

# training section
X, Y, ft = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
if opts.show_details:
    D, D0, S, S0, W, loss = train_details(D, D0, S, S0, W, X, Y, opts)
else:
    D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')
plot_result(X, Y, D, D0, S, S0, W, ft, loss, opts)

# testing section
X_test, Y_test, _ = load_toy(opts)
_, _, S, S0, _ = init(X_test, opts)
if opts.show_details:
    acc, y_hat = test_details(D, D0, S, S0, W, X_test, Y_test, opts)
else:
    acc, y_hat = test(D, D0, S, S0, W, X_test, Y_test, opts)
print('the test data accuracy is : ', acc)
plot_result(X_test, Y_test, D, D0, S, S0, W, ft, loss, opts)
plt.show()

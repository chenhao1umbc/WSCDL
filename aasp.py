"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
opts = OPT(C=16, K0=3, K=3, M=60)
opts.lamb = 0.1 # for sparsity penalty
opts.eta = 0 # for label penalty
opts.mu = 0  # for low rank penalty
# opts.show_details = False  # default as true

# training section
X, Y = load_data(opts)
D, D0, S, S0, W = init(X, opts)
D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
# plot_result(X, Y, D, D0, S, S0, W, ft=0, loss=loss, opts=opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
X_test, Y_test = load_data(opts, data='val')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc)
# plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss, opts=opts)
print('done')

"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
opts = OPT(C=16, K0=2, K=3, M=50)
opts.lamb, opts.eta, opts.mu = 0.1, 0.1, 0.01 # for sparsity penalty, label penalty, low rank penalty
opts.transpose, opts.shuffle, opts.show_details = False, False, False  # default as true

# training section
X, Y = load_data(opts)
D, D0, S, S0, W = init(X, opts)
D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)
plot_result(X, Y, D, D0, S, S0, W, ft=0, loss=loss, opts=opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
X_test, Y_test = load_data(opts, data='train')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc.acc)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss, opts=opts)
print('done')

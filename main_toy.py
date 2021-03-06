"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""
#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT()
opts.snr = 10
opts.lamb = 2.7 # for sparsity penalty
opts.eta = 3 # for label penalty
opts.mu = 0.1  # for low rank penalty
opts.show_details = False  # default as true

# training section
X, Y, opts.ft = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)
plot_result(X, Y, D, D0, S, S0, W, opts.ft, loss, opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
X_test, Y_test, _ = load_toy(opts, test='cv')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t, loss_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc.acc)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss_t, opts=opts)
print('done')

# %%

"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
opts = OPT()
opts.lamb = 0.1  # for sparsity penalty
opts.eta = 10  # for label penalty
opts.mu = 1  # for low rank penalty
X, Y, _ = load_toy(opts)
D, D0, S, S0, W = init(X, opts)

D, D0, S, S0, W, loss = train_beta(D, D0, S, S0, W, X, Y, opts)
# torch.save([D, D0, S, S0, W, opts, loss], '../DD0SS0Woptsloss'+tt().strftime("%y%m%d_%H_%M_%S")+'.pt')
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
X_test, Y_test, ft = load_toy(opts)
_, _, S, S0, _ = init(X, opts)
acc, y_hat = test_wscdl(D, D0, S, S0, W, X_test, Y_test, opts)
print('the test data accuracy is : ', acc)
plot_result(X_test, Y_test, D, D0, S, S0, W, ft, loss, opts)
plt.show()
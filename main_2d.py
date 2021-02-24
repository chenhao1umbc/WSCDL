"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""
#%%
from utils2 import *

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=16, K0=1, K=3)
# opts.dev = 'cpu'
opts.init, opts.shuffle, opts.show_details = 'rand', False, True
opts.Dh, opts.Dw, opts.batch_size = 128, 15, -1
opts.lamb, opts.lamb0, opts.eta, opts.mu = 1, 0.1, 1, 0.1 #sparsity, label, low rank

#%% training section
x, y, yy = load_data(opts, data='train')
D, D0, S, S0, W, loss = train(x, y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
plot_result(x, y, D, D0, S, S0, W, ft=0, loss=loss, opts=opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

#%% testing section, supposed that training is done
D, D0, S, S0, W, opts, loss = torch.load('../[5, 2, 9, 1, 0.1, 1]DD0SS0Woptsloss200506_02_07_37.pt', map_location='cpu')
X_test, Y_test, yy_test = load_data(opts, data='test')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t, loss_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy, recall and precision are : ', acc.acc, acc.recall, acc.f1)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss_t, opts=opts)
print('done')

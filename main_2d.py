"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""

from utils2 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=16, K0=1, K=3)
opts.init, opts.shuffle, opts.show_details = 'good', False, True
opts.Dh, opts.Dw, opts.batch_size = 256, 15, -1
opts.lamb, opts.eta, opts.mu = 1, 0.1, 0.01 # for sparsity penalty, label penalty, low rank penalty

# training section
X, Y = load_data(opts, data='train')
D, D0, S, S0, W, loss = train(X, Y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
plot_result(X, Y, D, D0, S, S0, W, ft=0, loss=loss, opts=opts)
# D, D0, S, S0, W, opts, loss = torch.load('DD0SS0Woptsloss.pt')

# testing section
D, D0, S, S0, W, opts, loss = torch.load('../[5, 2, 9, 1, 0.1, 1]DD0SS0Woptsloss200506_02_07_37.pt', map_location='cpu')
opts.dev = 'cpu'
X_test, Y_test = load_data(opts, data='val')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t, loss_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy, recall and precision are : ', acc.acc, acc.recall, acc.f1)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss_t, opts=opts)
print('done')

labels = 'alert,clearthroat,cough,doorknock,doorslam,drawer,keyboard,' \
         'keys,laughter,mouse,pageturn,pendrop,phone,printer,speech,switch'
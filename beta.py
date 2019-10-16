"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
opts = OPT()
opts.snr = 200
opts.lamb = 2 # for sparsity penalty
opts.eta = 0  # for label penalty
opts.mu = 0  # for low rank penalty
# opts.show_details = False  # default as true

# training section
X, Y, opts.ft = load_toy(opts)
D, D0, S, S0, W = init(X, opts)
D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)
plot_result(X, Y, D, D0, S, S0, W, opts.ft, loss, opts)

N, C = Y.shape
S_tik = torch.cat((S.mean(3), torch.ones(N, C, 1, device=S.device)), dim=-1)
exp_PtSnW = (S_tik * W).sum(2).exp()  # shape of [N, C]
exp_PtSnW[torch.isinf(exp_PtSnW)] = 1e38
y_hat = 1 / (1 + exp_PtSnW)
y_hat[y_hat > 0.5] = 1
y_hat[y_hat <= 0.5] = 0
label_diff = Y - y_hat
acc = label_diff[label_diff == 0].shape[0] / label_diff.numel()
print('The training data accuracy is : ', acc, '\n')
# plt.show()

# testing section
# opts.lamb = 5  # for sparsity penalty
X_test, Y_test, _ = load_toy(opts, test=True)
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc)
plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss, opts=opts)
print('done')

# # training section
# acc_av = 0
# X, Y, ft = load_toy(opts)
# for opts.lamb in range(20, 31):
#     opts.lamb= opts.lamb/10.0
#     for f in range(5):
#         D, D0, S, S0, W = init(X, opts)
#         D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
#         if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)
#
#         # testing section
#         X_test, Y_test, _ = load_toy(opts, test=True)
#         _, _, S_t, S0_t, _ = init(X_test, opts)
#         acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
#         acc_av = acc_av + acc
#     print('\nThe test data accuracy is : ', acc_av/5)
#     torch.save([acc_av/5, opts.lamb], '../acc_av' + tt().strftime("%y%m%d_%H_%M_%S") + '.pt')
#     acc_av = 0
# print('done')
"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
This file will sun aasp with multiple arguments in the terminal
this file is used for testing
"""

# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# opts = OPT(C=16, K0=3, K=3, M=80*3)
# opts.lamb = 0.1  # for sparsity penalty
# opts.eta = 0.001 # for label penalty
# opts.mu = 0.01 # for low rank penalty
# opts.transpose, opts.shuffle, opts.show_details = True, True, False  # default as true
# X, Y = load_data(opts)
# X_val, Y_val = load_data(opts, data='val')
#
# for i in [0.7, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]:
#     opts.lamb = i
#     # training section
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     D, D0, S, S0, W = init(X, opts)
#     D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
#     if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)
#
#     _, _, S_v, S0_v, _ = init(X_val, opts)
#     acc, y_hat, S_t, S0_t, loss_t = test(D, D0, S_v, S0_v, W, X_val, Y_val, opts)
#     print('\neta=0.001, mu =0.01, lambda =', i)
#     print('The test data accuracy is : ', acc.acc)
#     print('The test data recall is : ', acc.recall)
#     print('The test data precision is : ', acc.precision)
#
# # X_test, Y_test = load_data(opts, data='test')
# # _, _, S_t, S0_t, _ = init(X_test, opts)
# # acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
# # print('\nThe test data accuracy is : ', acc.acc)
# # print('\nThe test data recall is : ', acc.recall)
#
# # X_test, Y_test = load_data(opts, data='train')
# # _, _, S_t, S0_t, _ = init(X_test, opts)
# # acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
# # print('\nThe test data accuracy is : ', acc.acc)
# # print('\nThe test data recall is : ', acc.recall)
#
# # print((S==0).sum().item()/S.numel())
# # print((S_t==0).sum().item()/S_t.numel())
# # print((S0==0).sum().item()/S0.numel())
# # print((S0_t==0).sum().item()/S0_t.numel())
# # plot_result(X_test, Y_test, D, D0, S_t, S0_t, W, ft=0, loss=loss, opts=opts)

print('done')


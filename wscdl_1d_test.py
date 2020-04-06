"""This is the main fi%%%%%%setting model tunning parameters%%%%%%
le to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
This file will sun aasp with multiple arguments in the terminal
this file is used for testing
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
opts = OPT(C=16, K0=int(sys.argv[2]), K=3, M=50)
opts.lamb = 0.1  # for sparsity penalty
opts.eta = 0.01 # for label penalty
opts.mu = 0.01 # for low rank penalty
opts.transpose, opts.shuffle, opts.show_details = False, False, False  # default as true

# training section
for i in range(5):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X, Y = load_data(opts)
    D, D0, S, S0, W = init(X, opts)
    D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
    if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)

    # X_val, Y_val = load_data(opts, data='val')
    # _, _, S_v, S0_v, _ = init(X_val, opts)
    # acc, y_hat, S_v, S0_t, loss_v= test(D, D0, S_v, S0_v, W, X_val, Y_val, opts)
    # print('\nThe test data accuracy is : ', acc.acc)
    # print('\nThe test data recall is : ', acc.recall)

    X_test, Y_test = load_data(opts, data='test')
    _, _, S_t, S0_t, _ = init(X_test, opts)
    acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
    print('\nThe test data accuracy is : ', acc.acc)
    print('\nThe test data recall is : ', acc.recall)
print(vars(opts))
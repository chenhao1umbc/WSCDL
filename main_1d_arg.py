"""This is python -u aasp_v.py 2 0.05 0.001 0.01 > log_0.05_0.001_0.01.txtthe main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
This file will sun aasp with multiple arguments in the terminal
this file is used for tuning
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
opts = OPT(C=16, K0=3, K=3, M=80)
opts.lamb = float(sys.argv[2])  # for sparsity penalty
opts.eta = float(sys.argv[3]) # for label penalty
opts.mu = float(sys.argv[4])  # for low rank penalty
opts.transpose, opts.shuffle, opts.show_details = True, True, False  # default as true

# training section
result1, result2 = [], []
for i in range(5):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X, Y = load_data(opts)
    D, D0, S, S0, W = init(X, opts)
    D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
    if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)

    X_val, Y_val = load_data(opts, data='val')
    _, _, S_v, S0_v, _ = init(X_val, opts)
    acc, y_hat, S_v, S0_t, loss_v= test(D, D0, S_v, S0_v, W, X_val, Y_val, opts)
    print('\nThe validation data current accuracy is : ', acc.acc)
    print('\nThe validation data current recall is : ', acc.recall)
    result1.append(acc.acc)
    result2.append(acc.recall)

    # X_test, Y_test = load_data(opts, data='test')
    # _, _, S_t, S0_t, _ = init(X_test, opts)
    # acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
    # print('\nThe test data accuracy is : ', acc.acc)
    # print('\nThe test data recall is : ', acc.recall)
print('averaged mean, recall', sum(result1)/5, sum(result2)/5)
print(vars(opts))

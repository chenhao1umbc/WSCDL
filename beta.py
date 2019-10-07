"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
opts = OPT()
opts.snr = 200
opts.lamb = 2.7 # for sparsity penalty
opts.eta = 4.1 # for label penalty
opts.mu = 0  # for low rank penalty
opts.show_details = False  # default as true
acc_av = 0.0

# training section
X, Y, ft = load_toy(opts)
for opts.eta in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.85, 4.1]:
    for f in range(5):
        D, D0, S, S0, W = init(X, opts)
        D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
        if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)

        # testing section
        X_test, Y_test, _ = load_toy(opts, test=True)
        _, _, S_t, S0_t, _ = init(X_test, opts)
        acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
        acc_av = acc_av + acc
    print('\nThe test data accuracy is : ', acc_av/5)
    acc_av = 0
    torch.save([acc_av, opts.eta], '../acc_av' + tt().strftime("%y%m%d_%H_%M_%S") + '.pt')
print('done')
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
opts = OPT(C=16, K0=3, K=3, M=30)
opts.lamb = float(sys.argv[1])  # for sparsity penalty
opts.eta = 0 # for label penalty
opts.mu = 0  # for low rank penalty
# opts.show_details = False  # default as true

# training section
X, Y = load_data(opts)
D, D0, S, S0, W = init(X, opts)
D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
if opts.save_results: save_results(D, D0, S, S0, W, opts, loss)

X_test, Y_test, _ = load_data(opts, data='val')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t = test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc)
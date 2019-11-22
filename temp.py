# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# opts = OPT()
# opts.snr = 10
# opts.lamb = 2.7 # for sparsity penalty
# opts.eta = 3 # for label penalty
# opts.mu = 0  # for low rank penalty
# opts.show_details = True  # default as true
#
# # training section
# X, Y, opts.ft = load_toy(opts)
# # D, D0, S, S0, W = init(X, opts)
# # D, D0, S, S0, W, loss = train(D, D0, S, S0, W, X, Y, opts)
# # if opts.save_results*0: save_results(D, D0, S, S0, W, opts, loss)
# # plot_result(X, Y, D, D0, S, S0, W, opts.ft, loss, opts)
# D, D0, S, S0, W, opts, loss = torch.load('./../[2.7, 3, 0]DD0SS0Woptsloss191122_08_58_12.pt')
#
# # testing section
# X_test, Y_test, _ = load_toy(opts, test='cv')
# _, _, S_t, S0_t, _ = init(X_test, opts)
# opts.lamb2 = 1
# acc1, y_hat, s, S0_t, loss1 = test_fista(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
# acc2, y_hat, ss, S0_t, loss2 = test_fista(D, D0, torch.rand(S_t.shape, device=opts.dev), S0_t, W, X_test, Y_test, opts)
# print('\nloss1', loss1[0], loss1[-1])
# print('loss2', loss2[0], loss2[-1])
# sf, ssf = s.flatten(), ss.flatten()
# a, b = torch.zeros(sf.numel()), torch.zeros(ssf.numel())
# a[sf!=0] = 1
# b[ssf!=0] = 1
# print('support diff',a.sum(), b.sum(), (a-b).abs().sum())
# print('done')



# ###########################################################

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
[D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
opts.lamb2 = 0.01
ind = torch.randperm(D.numel())
d = D.flatten()[ind].reshape(D.shape)
X, Y = load_data(opts)
X_test, Y_test = load_data(opts, data='train')
_, _, S_t, S0_t, _ = init(X_test, opts)
opts.show_details = False  # default as true

X_test, Y_test = X_test[:2], Y_test[:2]
acc1, y_hat1, s, S0_t1, loss1= test(D, D0, S[:2], S0[:2], W, X_test, Y_test, opts)
acc2, y_hat2, ss, S0_t2, loss2 = test(D, D0, awgn(S[:2], 10), awgn(S0[:2],10), W, X_test, Y_test, opts)
# acc0, y_hat, ss, S0_t, loss_t = test(D, D0, torch.rand(S_t.shape, device=opts.dev), S0_t, W, X_test, Y_test, opts)
print('\nloss1', loss1[0], loss1[-1])
print('loss2', loss2[0], loss2[-1])
sf, ssf = s.flatten(), ss.flatten()
a, b = torch.zeros(sf.numel()), torch.zeros(ssf.numel())
a[sf!=0] = 1
b[ssf!=0] = 1
print('support diff',a.sum(), b.sum(), (a-b).abs().sum())
print('done')

# ###########################################################
# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# [D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
# opts.lamb2 = 0.01
# X, Y = load_data(opts)
# X_test, Y_test = load_data(opts, data='train')
# _, _, S_t, S0_t, _ = init(X_test, opts)
# acc, y_hat, S_t, S0_t, loss_t= test_fista(D, D0, S, S0, W, X_test, Y_test, opts)
# # acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S, S0, W, X_test, Y_test, opts)
# print('\nThe test data accuracy is : ', acc.acc)
# print('\nThe test dasata recall is : ', acc.recall)

# ###########################################################
# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 3'
# [D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
# X, Y = load_data(opts)
# X_test, Y_test = load_data(opts, data='train')
# x = X.t().cpu()
# N, C, K, T = S.shape
# d = torch.tensor([])
# for c, k in [(i, j) for i in range(C) for j in range(K)]:
#     dck = D[c, k, :].cpu()  # shape of [M]
#     Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]
#     d = torch.cat((d, Tdck), 1)
# Tdck0 = (toeplitz(D0, m=T, T=T).squeeze()).t().cpu()  # shape of [T, m=T]
# d = torch.cat((d, Tdck0), 1)
#
# s = S.reshape(N, -1).t().cpu()
# s = torch.cat((s, S0.squeeze().t().cpu()))
#
# k =  d.shape[1]  # or k = 2000 tog get fast result
# dd = d[:, :k]
# n = 300
#
# # spam output
# alpha = spams.lasso(np.asfortranarray(x.numpy()[:,:n]), D=np.asfortranarray(dd.numpy()), lambda1=opts.lamb/4)
# a = np.asarray(sparse.csc_matrix.todense(alpha))  # shape of [C*K*T + k0*T, N]
#
# # BPGM output
# # dd = dd.cuda()
# maxiter, correction, threshold = 5000, 0.7, 1e-4
# Mw = opts.delta * correction  # correction is help to make the loss monotonically decreasing
# lamb = opts.lamb
# # 'skc update will lead sc change'
# abs_Tdck = abs(dd)
# Tdck_t_Tdck = dd.t() @ dd  # shape of [CKT, CKT]
# Tdckt_bt = dd.t() @ x[:,:n]  # shape of [CKT, N]
# M = (abs_Tdck.t() @ abs_Tdck + 1e-38).sum(1)  # M is the diagonal of majorization matrix, shape of [T]
#
# ss  = s[:k,:n]
# ss = ss -ss
# sn, snold = ss.clone(), ss.clone()  # shape of [CKT, N]
# for i in range(maxiter):
#     sn_til = sn + Mw * (sn - snold)  # shape of [CKT, N]
#     nu = sn_til.t() - (Tdck_t_Tdck @ sn_til - Tdckt_bt).t() / M  # shape of [N, T]
#     snnew = shrink(M, nu, lamb / 2).t()  # shape of [T, N]
#     snold, sn = sn, snnew  # make sure sc is updated in each loop
#     if torch.norm(sn - snold) / (sn.norm() + 1e-38) < threshold: break
#     torch.cuda.empty_cache()
# print('used iters :', i)
#
# # # fista output  -- it is really slow when k = 196000, more than 2000 iterations
# # so = fista(x.t()[:n,:], dd, torch.zeros(k, n).t(), opts.lamb).t()  # shape of [C*K*T + k0*T, N]
#
# plt.plot(a.flatten(), 'o')
# # plt.plot(so.squeeze().numpy().flatten(), 'x')
# plt.plot(snnew.squeeze().numpy().flatten(), '*')


# ###########################################################
# # method 2 overall lasso
# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# [D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
# X, Y = load_data(opts)
# X_test, Y_test = load_data(opts, data='train')
# x = X.t().cpu()
# N, C, K, T = S.shape
# d = torch.tensor([])
# for c, k in [(i, j) for i in range(C) for j in range(K)]:
#     dck = D[c, k, :].cpu()  # shape of [M]
#     Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]
#     d = torch.cat((d, Tdck), 1)
# Tdck0 = (toeplitz(D0, m=T, T=T).squeeze()).t().cpu()  # shape of [T, m=T]
# d = torch.cat((d, Tdck0), 1)
#
# s = S.reshape(N, -1).t().cpu()
# s = torch.cat((s, S0.squeeze().t().cpu()))
#
# k =  d.shape[1]  # or k = 2000 tog get fast result
# dd = d[:, :k]
# n = 300
#
# # spam output
# print('running')
# # for i in [ 20, 30, 50, 80, 100, 150, 200 ]:
# for i in [ 1000, 3000, 5000, 100000, 200000 ]:
#     alpha = spams.lasso(np.asfortranarray(x.numpy()[:,:n]), D=np.asfortranarray(dd.numpy()), lambda1=opts.lamb/i)
#     a = np.asarray(sparse.csc_matrix.todense(alpha))  # shape of [C*K*T + k0*T, N]
#     ss = torch.tensor(a[:192000,:]).reshape(C, K, T, N).permute(3,0,1,2).cuda()
#     get_perf(Y, ss, W)
#     print('0s percentage :', (ss==0).sum().item()/ss.numel())
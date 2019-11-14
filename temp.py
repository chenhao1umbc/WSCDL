# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 3'
# [D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
# X, Y = load_data(opts)
# X_test, Y_test = load_data(opts, data='train')
# _, _, S_t, S0_t, _ = init(X_test, opts)
# acc, y_hat, S_t, S0_t, loss_t= test_fista(D, D0, S, S0, W, X_test, Y_test, opts)
# # acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
# print('\nThe test data accuracy is : ', acc.acc)
# print('\nThe test dasata recall is : ', acc.recall)

###########################################################
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 3'
[D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
X, Y = load_data(opts)
X_test, Y_test = load_data(opts, data='train')
x = X.t().cpu()
N, C, K, T = S.shape
d = torch.tensor([])
for c, k in [(i, j) for i in range(C) for j in range(K)]:
    dck = D[c, k, :].cpu()  # shape of [M]
    Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]
    d = torch.cat((d, Tdck), 1)
Tdck0 = (toeplitz(D0, m=T, T=T).squeeze()).t().cpu()  # shape of [T, m=T]
d = torch.cat((d, Tdck0), 1)

s = S.reshape(N, -1).t().cpu()
s = torch.cat((s, S0.squeeze().t().cpu()))

k =  d.shape[1]  # or k = 2000 tog get fast result
dd = d[:, :k]
n = 300

# spam output
alpha = spams.lasso(np.asfortranarray(x.numpy()[:,:n]), D=np.asfortranarray(dd.numpy()), lambda1=opts.lamb/4)
a = np.asarray(sparse.csc_matrix.todense(alpha))  # shape of [C*K*T + k0*T, N]

# BPGM output
# dd = dd.cuda()
maxiter, correction, threshold = 5000, 0.7, 1e-4
Mw = opts.delta * correction  # correction is help to make the loss monotonically decreasing
lamb = opts.lamb
# 'skc update will lead sc change'
abs_Tdck = abs(dd)
Tdck_t_Tdck = dd.t() @ dd  # shape of [CKT, CKT]
Tdckt_bt = dd.t() @ x[:,:n]  # shape of [CKT, N]
M = (abs_Tdck.t() @ abs_Tdck + 1e-38).sum(1)  # M is the diagonal of majorization matrix, shape of [T]

ss  = s[:k,:n]
ss = ss -ss
sn, snold = ss.clone(), ss.clone()  # shape of [CKT, N]
for i in range(maxiter):
    sn_til = sn + Mw * (sn - snold)  # shape of [CKT, N]
    nu = sn_til.t() - (Tdck_t_Tdck @ sn_til - Tdckt_bt).t() / M  # shape of [N, T]
    snnew = shrink(M, nu, lamb / 2).t()  # shape of [T, N]
    snold, sn = sn, snnew  # make sure sc is updated in each loop
    if torch.norm(sn - snold) / (sn.norm() + 1e-38) < threshold: break
    torch.cuda.empty_cache()
print('used iters :', i)

# # fista output  -- it is really slow when k = 196000, more than 2000 iterations
# so = fista(x.t()[:n,:], dd, torch.zeros(k, n).t(), opts.lamb).t()  # shape of [C*K*T + k0*T, N]

plt.plot(a.flatten(), 'o')
# plt.plot(so.squeeze().numpy().flatten(), 'x')
plt.plot(snnew.squeeze().numpy().flatten(), '*')

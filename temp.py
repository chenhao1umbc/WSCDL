from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
[D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
X, Y = load_data(opts)
X_test, Y_test = load_data(opts, data='train')



# x = X.t().cpu()
#
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
# k = d.shape[1]
# dd = d[:, :k]
# n = 1
#
# alpha = spams.lasso(np.asfortranarray(x.numpy()[:,:n]), D=np.asfortranarray(dd.numpy()), lambda1=opts.lamb/2)
# a = np.asarray(sparse.csc_matrix.todense(alpha))  # shape of [C*K*T + k0*T, N]
#
# so = fista(x.t()[:n,:], dd, torch.zeros(k, n).t(), opts.lamb).t()  # shape of [C*K*T + k0*T, N]


_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t, loss_t= test_fista(D, D0, S, S0, W, X_test, Y_test, opts)
# acc, y_hat, S_t, S0_t, loss_t= test(D, D0, S_t, S0_t, W, X_test, Y_test, opts)
print('\nThe test data accuracy is : ', acc.acc)
print('\nThe test dasata recall is : ', acc.recall)



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

k = 20000 # d.shape[1]
dd = d[:, :k]
n = 1

alpha = spams.lasso(np.asfortranarray(x.numpy()[:,:n]), D=np.asfortranarray(dd.numpy()), lambda1=opts.lamb/2)
a = np.asarray(sparse.csc_matrix.todense(alpha))  # shape of [C*K*T + k0*T, N]

so = fista(x.t()[:n,:], dd, torch.zeros(k, n).t(), opts.lamb).t()  # shape of [C*K*T + k0*T, N]
plt.plot(a.flatten(), 'o')
plt.plot(so.squeeze().numpy().flatten(), 'x')



maxiter, correction, threshold = 500, 0.7, 1e-4
Mw = opts.delta * correction  # correction is help to make the loss monotonically decreasing
lamb = opts.lamb
dev = opts.dev
P = torch.ones(1, T, device=dev) / T  # shape of [1, T]
# 'skc update will lead sc change'
abs_Tdck = abs(d)
Tdck_t_Tdck = d.t() @ d  # shape of [CKT, CKT]
Tdckt_bt = d.t() @ x  # shape of [CKT, N]
M = (abs_Tdck.t() @ abs_Tdck + 1e-38).sum(1)  # M is the diagonal of majorization matrix, shape of [T]
sn, snold = s.clone(), s.clone()  # shape of [CKT, N]


for i in range(maxiter):
    sn_til = sn + Mw * (sn - snold)  # shape of [CKT, N]
    nu = sn_til - (Tdck_t_Tdck @ sn_til - Tdckt_bt).t() / M  # shape of [N, T]
    sck_new = shrink(M, nu, lamb / 2)  # shape of [N, T]
    sck_old[:], sck[:] = sck[:], sck_new[:]  # make sure sc is updated in each loop
    if torch.norm(sck - sck_old) / (sck.norm() + 1e-38) < threshold: break
    torch.cuda.empty_cache()
print('M max', M.max())
if marker == 1:
    print('--inf to 1e38 happend within the loop')
    plt.figure();
    plt.plot(loss.cpu().numpy(), '-x')
    print('How many inf to 1e38 happend finally', exp_PtSnc_tilWc[exp_PtSnc_tilWc == 1e38].shape[0])
if (loss[0] - loss[-1]) < 0:
    wait = input("Loss Increases, PRESS ENTER TO CONTINUE.")
print('sck loss after bpgm the diff is :%1.9e' % (loss[0] - loss[-1]))
plt.figure();
plt.plot(loss.cpu().numpy(), '-x')
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
[D, D0, S, S0, W, opts, loss] = torch.load('/home/chenhao1/Hpython/[0.1, 0.01, 0.01]DD0SS0Woptsloss191106_10_50_30.pt')
X, Y = load_data(opts)
X_test, Y_test = load_data(opts, data='train')
_, _, S_t, S0_t, _ = init(X_test, opts)
acc, y_hat, S_t, S0_t, loss_t= test_fista(D, D0, S, S0_t, W, X_test, Y_test, opts)
# acc, y_hat, S_t, S0_t, loss_t= test_fista(D.double(), D0.double(), (S-S).double(), S0_t.double(), W.double(), X_test.double(), Y_test, opts)
print('\nThe test data accuracy is : ', acc.acc)
print('\nThe test data recall is : ', acc.recall)

N, C, K, T = S.shape
d = torch.tensor([], device=S.device)
for c, k in [(i, j) for i in range(C) for j in range(K)]:
    dck = D[c, k, :]  # shape of [M]
    Tdck = (toeplitz(dck.unsqueeze(0), m=T, T=T).squeeze()).t()  # shape of [T, m=T]
    d = torch.cat((d, Tdck), 1)
Tdck0 = (toeplitz(D0, m=T, T=T).squeeze()).t()  # shape of [T, m=T]
d = torch.cat((d, Tdck0), 1)

s = S.permute(3,0,1,2).reshape(N, -1).t()
s = torch.cat((s, S0.squeeze().t()))
"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""
#%%
from utils2 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=10, K0=1, K=2)
opts.init, opts.shuffle, opts.show_details = 'rand', False, True
opts.Dh, opts.Dw, opts.batch_size = 100, 25, -1
opts.lamb, opts.lamb0, opts.eta, opts.mu = 0.1, 0.1, 1, 0.1 #sparsity, label, low rank

#%% training section
X, Y, yy = load_data(opts, data='train') # shape of [n_sample, f, t]
D, D0, S, S0, W, loss = train(X, Y, opts)

#%% visualize learned atoms
for i in range(10):
    fig= plt.figure()
    fig.set_size_inches(w=12, h=8)
    d = D[i].permute(0,2,1).reshape(opts.Dh, opts.K*opts.Dw).cpu()
    plt.imshow(d, aspect='auto', interpolation='None')
    plt.title(f'Class {i} atoms')


#%% testing section
X_val, Y_val, yy_val = load_data(opts, data='val')
_, _, St, S0t, _ = init(X_val, opts)
acc, y_hat, St, S0t, loss_t = test(D, D0, St, S0t, W, X_val, Y_val, opts)
print('The validation accuracy, recall and precision are : ',\
            acc.acc, acc.recall, acc.f1)


# %% check results
fig= plt.figure()
fig.set_size_inches(w=12, h=8)
plt.imshow(y_hat.cpu(),aspect='auto', interpolation='None')
plt.title('y_hat')

fig= plt.figure()
fig.set_size_inches(w=12, h=8)
yt = y_hat.clone()
thr = 0.3
yt[yt>=thr] = 1
yt[yt<thr] = 0
plt.imshow(yt.cpu(),aspect='auto', interpolation='None')
plt.title(f'y_hat with threshold {thr}')

fig= plt.figure()
fig.set_size_inches(w=12, h=8)
plt.imshow(Y_val.cpu(),aspect='auto', interpolation='None')
plt.title('Y')

# %%

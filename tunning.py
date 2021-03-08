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

# %% analysis result
record = torch.load('tunning.pt')
n = len(record)
res = torch.rand(n, 3)
param = torch.rand(n,5)
for i,v in enumerate(record):
    res[i] = torch.tensor(v[0])
    param[i] = torch.tensor(v[1])
value, index = res.max(0)
print('max vlaues :', value)

for i in index:
    print([i], param[i])

               
# a function of given parameters to return the result tensors
def get_result(res, param, Dw=0, lamb=0, lamb0=0, eta=0, mu=0):
    """ if Dw, lamb etc. is 0, that means coresponding column are all selected
        otherwise Dw etc. is a value from its pool,e.g.
        pool_Dw = [ 7., 15., 21., 25., 29., 35., 45.]
        Dw = 7. or Dw=15.

        param has the shape of [n_record, 5]
        each of the 5 columns means [Dw, lamb, lamb0, eta, mu]

        res has the shape of [[n_record, 5]]
        each of the 3 columns means [acc, recall, F1]
    """
    # find the indecies of given param
    n = param.shape[0]
    idx = torch.arange(n)
    if Dw!=0 : 
        res_ind = idx[param[:,0] == Dw]
    else:
        res_ind = idx.clone()

    if lamb!=0: res_ind = np.intersect1d(idx[param[:,1] == lamb], res_ind)
    if lamb0 !=0: res_ind = np.intersect1d(idx[param[:,2] == lamb0], res_ind)
    if eta !=0: res_ind = np.intersect1d(idx[param[:,3] == eta], res_ind)
    if mu !=0: res_ind = np.intersect1d(idx[param[:,4] == mu], res_ind)

    return res[res_ind], res_ind

r, idx = get_result(res, param, Dw=0, lamb=0.1, lamb0=0.1, eta=0.01, mu=0.1)
print(r)
print(param[idx])

#TODO lamb0 is and eta are on the boundary, lamb0 gets larger, eta gets smaller to see the result
# %% compare with others' result
route = '/home/chenhao1/Matlab/WSCDL/'
res = sio.loadmat(route+'res_knn.mat')
res = res['Pre_Labels']
res[res==-1]=0
res = res.T
metrics.f1_score(Y_test.cpu().flatten(), res.flatten())
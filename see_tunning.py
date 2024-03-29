"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""
#%%
from utils2 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=10, K0=1, K=2)
opts.init, opts.shuffle, opts.show_details = 'rand', False, True
opts.Dh, opts.Dw, opts.batch_size = 100, 29, -1
opts.lamb, opts.lamb0, opts.eta, opts.mu = 0.1, 0.1, 0.01, 0.1 #sparsity, label, low rank

# %% analysis result
record = torch.load('tunning_rough.pt')
n = len(record)
res = torch.rand(n, 3)
param = torch.rand(n,5)
for i,v in enumerate(record):
    res[i] = torch.tensor(v[0])  # [n, acc, recall, f1]
    param[i] = torch.tensor(v[1]) # [n, Dw, lamb, lamb0, eta, mu]
value, index = res.max(0)
print('max acc, recall, f1, vlaues :', value, '\n')

for i, v in enumerate(index):
    print(f"max {['acc', 'recall', 'f1'][i]} index and vlaues :", res[v])
    print([v], param[v], '\n')

               
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
"Dw=29, lamb=0.1, lamb0=0.1, eta=0.01, mu=0.1"
"Dw=21, lamb=0.1, lamb0=1, eta=0.001, mu=1"
"Dw=21, lamb=0.1, lamb0=0.1, eta=0.001, mu=1"
r, idx = get_result(res, param, Dw=21, lamb=0.1, lamb0=0.1, eta=0.001, mu=1)
print(r)
print(param[idx])

for i in range(5):
    if param[idx][:, i].unique().shape[0] >1 :
        which_is_0 = i
        break
fig = plt.figure()
fig.set_size_inches(w=6, h=4)
index = param[idx][:, which_is_0].sort()[1]
plt.plot(param[idx][:, which_is_0].sort()[0], r[:, -1][index], '-x')


#%%
record = torch.load('tunning.pt')
res = torch.zeros(80, 3)
para = torch.zeros(80, 5)
for i,r in enumerate(record):
    res[i] = torch.tensor(r[0])
    para[i] = torch.tensor(r[1])

res = res.reshape(5, 16, -1).mean(0)
para = para.reshape(5, 16, -1).mean(0)
v, i = res.sort()
"Dw=29, lamb=0.1, lamb0=0.1, eta=0.001, mu=0.1 is the best" 

# %% compare with others' result
route = '/home/chenhao1/Matlab/WSCDL/'
# res = sio.loadmat(route+'res_knn.mat')
# res = res['Pre_Labels']
# res[res==-1]=0
# res = res.T
# metrics.f1_score(Y_test.cpu().flatten(), res.flatten())

with open(route+'you_raich_0.txt') as f:
    data =f.readlines()
rec = []
prec = []
count = 0
for i, d in enumerate(data):
    if d == 'rec =\n':
        rec.append(float(data[i+2][4:10]))
        count += 1

    if d == 'prec =\n':
        prec.append(float(data[i+2][4:10]))
        if count == 10:
            print(rec[-1])
            print(prec[-1])

rec, prec = torch.tensor(rec), torch.tensor(prec)
f1 = 2/(1/rec+1/prec)
v, i = f1.sort()
# best lamb, winzize, N
# 10, 30, 200
# 10, 100, 10
# 10, 50, 50



#%% visualize learned atoms
param = str([opts.K, opts.K0, opts.Dw, opts.lamb, opts.lamb0, opts.eta , opts.mu])
D, D0, S, S0, W, opts, loss = \
    torch.load('../saved_dicts/'+param+'DD0SS0Woptsloss.pt', map_location='cpu')

for i in range(10):
    fig= plt.figure()
    fig.set_size_inches(w=4, h=6)
    d = D[i].permute(0,2,1).reshape(opts.Dh, opts.K*opts.Dw).cpu()
    plt.imshow(d, aspect='auto', interpolation='None')
    plt.title(f'Class {i} atoms')






# %%
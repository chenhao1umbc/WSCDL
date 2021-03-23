"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
This file is used for training and validation

labels = 'alert,clearthroat,cough,doorknock,doorslam,drawer,keyboard,' \
         'keys,laughter,mouse,pageturn,pendrop,phone,printer,speech,switch'
"""
#%%
from utils2 import *

#%% Init parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=10, K0=1, K=2)
opts.init, opts.shuffle, opts.show_details = 'rand', True, False
opts.Dh, opts.Dw, opts.batch_size = 100, 25, -1
opts.lamb, opts.lamb0, opts.eta, opts.mu = 1, 0.1, 1, 0.1 #sparsity, label, low rank

#%% load data
X, Y, yy = load_data(opts, data='train') # shape of [n_sample, f, t]
X_val, Y_val, yy_val = load_data(opts, data='val')

# #%% training section first run
# res = []
# for Dw in [15, 21, 25, 29, 35]:
#     for lamb in [0.1, 0.01, 0.001, 0.5]:
#         for lamb_ratio in [1, 0.5, 0.1 ]:
#             for eta in [0.1, 0.01, 1]:
#                 for mu in [0.1, 0.01, 1]:
#                     opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu = \
#                                         Dw, lamb, lamb_ratio*lamb, eta, mu 
#                     D, D0, S, S0, W, loss = train(X, Y, opts)
#                     save_results(D, D0, S, S0, W, opts, loss)

#                     _, _, St, S0t, _ = init(X_val, opts)
#                     acc, y_hat, St, S0t, loss_t = test(D, D0, St, S0t, W, X_val, Y_val, opts)
#                     print('The validation accuracy, recall and precision are : ',\
#                              acc.acc, acc.recall, acc.f1)
#                     print('with parameters as : ', opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)
#                     print('\n')
#                     res.append(((acc.acc, acc.recall, acc.f1),(opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)))
#                     torch.save(res, 'tunning.pt')


#%% training section 
# lamb0 is and eta are on the boundary, lamb0 gets larger, eta gets smaller to see the result
# res = [] 
# for Dw in [21]:
#     for lamb in [0.1, 0.01, 0.001, 0.5]:
#         for lamb_ratio in [1, 0.5, 0.1, 10]:
#             for eta in [0.001, 0.0001]:
#                 for mu in [10, 1]:
#                     opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu = \
#                                         Dw, lamb, lamb_ratio*lamb, eta, mu 
#                     D, D0, S, S0, W, loss = train(X, Y, opts)
#                     save_results(D, D0, S, S0, W, opts, loss)

#                     res = torch.load('tunning.pt')
#                     _, _, St, S0t, _ = init(X_val, opts)
#                     acc, y_hat, St, S0t, loss_t = test(D, D0, St, S0t, W, X_val, Y_val, opts)
#                     print('The validation accuracy, recall and precision are : ',\
#                              acc.acc, acc.recall, acc.f1)
#                     print('with parameters as : ', opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)
#                     print('\n')
#                     res.append(((acc.acc, acc.recall, acc.f1),(opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)))
#                     torch.save(res, 'tunning.pt')

#%% after rough tunning the followings are the best ones
"Dw=29, lamb=0.1, lamb0=0.1, eta=0.01, mu=0.1"
"Dw=21, lamb=0.1, lamb0=1, eta=0.001, mu=1"
"Dw=21, lamb=0.1, lamb0=0.1, eta=0.001, mu=1"
res = []

for runs in range(5):
    X, Y, yy = load_data(opts, data='train') # shape of [n_sample, f, t]
    X_val, Y_val, yy_val = load_data(opts, data='val')
    for Dw in [21, 29]:
        for lamb in [0.1, 0.01]:
            for lamb_ratio in [1, 0.1]:
                for eta in [0.001]:
                    for mu in [0.1, 1]:
                        opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu = \
                                            Dw, lamb, lamb_ratio*lamb, eta, mu 
                        D, D0, S, S0, W, loss = train(X, Y, opts)
                        save_results(D, D0, S, S0, W, opts, loss)

                        # res = torch.load('tunning.pt')
                        _, _, St, S0t, _ = init(X_val, opts)
                        acc, y_hat, St, S0t, loss_t = test(D, D0, St, S0t, W, X_val, Y_val, opts)
                        print('The validation accuracy, recall and precision are : ',\
                                acc.acc, acc.recall, acc.f1)
                        print('with parameters as : ', opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)
                        print('\n')
                        res.append(((acc.acc, acc.recall, acc.f1),(opts.Dw, opts.lamb, opts.lamb0, opts.eta, opts.mu)))
                        torch.save(res, 'tunning.pt')

#%% just validation section, supposed that training is done
D, D0, S, S0, W, opts, loss = torch.load(
    '../saved_dicts/[3, 1, 5, 0.1, 0.01, 1, 0.1]DD0SS0Woptsloss.pt')
X_val, Y_val = load_data(opts, data='val')
_, _, St, S0t, _ = init(X_val, opts, init='rand')
acc, y_hat, St, S0t, loss_t = test(D, D0, St, S0t, W, X_val, Y_val, opts)
print('\nThe test data accuracy, recall and precision are : ', acc.acc, acc.recall, acc.f1)
plot_result(X_val, Y_val, D, D0, St, S0t, W, ft=0, loss=loss_t, opts=opts)
print('done')

# %%

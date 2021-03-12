"""
This function is made to calculate convolution for matlab to call
You_raich conv is too slow. matlab conv 1d only support vectors, to many for loop for tensor calculation
In addition, matlab gpu conv is much slower than cpu conv
So this file will make the tensor convoluiton faster without for loop, with/out GPU
"""
#%%
import torch
import torch.nn.functional as Func
def conv(x, ker):
    if torch.cuda.is_available():
        #x = x.cuda()  # shape of [ F, T, n_batch]
        ker = ker.cuda() # shape of [ F, win_size, n_kernels]
    x = x.permute(1,2,0)[:,None] # shape [n_batch, 1,  F, T]
    ker = ker.permute(1,2,0)[:, None]  # shape of [n_kernels, 1, F, win_size]
    # res has shape of [n_batch, n_kernels, F, T]
    res = Func.conv2d(x, ker, padding=(0, ker.shape[-1]//2)) 

    return res.cpu()

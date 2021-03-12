"""
This function is made to calculate convolution for matlab to call
You_raich conv is too slow. matlab conv 1d only support vectors, to many for loop for tensor calculation
In addition, matlab gpu conv is much slower than cpu conv
So this file will make the tensor convoluiton faster without for loop, with/out GPU
"""
#%%
import torch
import torch.nn.functional as Func
torch.backends.cudnn.deterministic = True

def conv(x, ker):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ker = ker.cuda() # shape of [ F, win_size, n_kernels]
    
    x = x[:, None] # x shape [n_batch, 1,  F, T]
    ker = ker.permute(2,0,1)[:, None]  # shape of [n_kernels, 1, F, win_size]

    # res has shape of [n_batch, n_kernels, 1, T+padding]
    res = Func.conv2d(x, ker.flip([3]), padding=(0, (ker.shape[-1]-1))) 

    return res.T.squeeze().cpu().numpy()

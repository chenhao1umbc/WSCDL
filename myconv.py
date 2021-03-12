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
    if torch.cuda._is_availabel():
        x = x.cuda()
        ker = ker.cuda()
    res = Func.conv1d(x, ker, padding=ker.shape[-1]//2, groups=1)

    return res.cpu()
'''This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
'''

from utils import *

opts = OPT()


'''demo'''
print('CPU version')
a = torch.rand(300, 5, 500) # S0
b = torch.rand(5, 40)  # D0
k0, m = b.shape
t = time.time()
s0 = a.unsqueeze(1).unsqueeze(1)  # expand dimension for conv1d
d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
for i in range(300):
    for ii in range(k0):
        r1 = F.conv1d(s0[i,:, :, ii, :], d0[:, :, ii, :])
print('Double for-loop time is :', time.time()-t )
t = time.time()
s0 = a.unsqueeze(1)  # expand dimension for conv1d
d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
for i in range(k0):
    r2= F.conv1d(s0[:, :, i, :], d0[:, :, i, :])
print('Single for-loop time is :', time.time()-t)
'compare wth'
t = time.time()
r2 = F.conv1d(a, b.flip(1).unsqueeze(1), groups=k0)
print('Without for-loop time is:', time.time()-t)
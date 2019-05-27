from scipy.io import loadmat# for matlab mat 7.0 , 7.1 , 7.2
import h5py  # for matlab mat 7.3
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
         # zeros([5,5])# from mpl_toolkits.mplot3d import Axes3D

"""here is the rename of some functions because it is too long and often used"""
'''----->'''
conv = np.convolve
fft = np.fft.fft
fft2 = np.fft.fft2
ifft = np.fft.ifft
norm = np.linalg.norm
np.set_printoptions(linewidth=200)
ones = np.ones              # ones([5,5])
pinv = np.linalg.pinv
rand = np.random.rand
randn = np.random.randn
rank = np.linalg.matrix_rank
svd = np.linalg.svd
zeros = np.zeros            # zeros([5,5])
'''<-----'''
"""here is the rename of some functions because it is too long and often used"""

def updateD(D, S, X, opts):
    '''this function is to update the distinctive D using BPG-M
    input is initialed D,
    sparse coeffecient S
    training Data X    
    training paremeters opts
    output is updated D'''
    return D

def fconv(s1=1, s2=1):
    '''this function is using fft to do convolution for 1d data'''
    n = len(s1) + len(s2) -1
    return ifft(fft(s1, n) * fft(s2, n))

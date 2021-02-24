"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32
"""
#%%
from utils2 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=16, K0=1, K=3)
opts.init, opts.shuffle, opts.show_details = 'good', False, True
opts.Dh, opts.Dw, opts.batch_size = 256, 15, 3
opts.lamb, opts.eta, opts.mu = 1, 0.1, 0.01 # for sparsity penalty, label penalty, low rank penalty

#%% training section
tr = load_data(opts, data='train')


#%% testing section



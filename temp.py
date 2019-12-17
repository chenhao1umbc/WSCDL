from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
opts = OPT(C=16, K0=3, K=3, M=80*3)
opts.lamb = 0.1  # for sparsity penalty
opts.eta = 0.001 # for label penalty
opts.mu = 0.01 # for low rank penalty
opts.transpose, opts.shuffle, opts.show_details = False, False, False  # default as true
X, Y = load_data(opts)
X_val, Y_val = load_data(opts, data='val')

label = ['alert', 'clearthroat', 'cough', 'doorknock', 'doorslam', 'drawer', 'keyboard', 'keys', 'laughter',
 'mouse', 'pageturn', 'pendrop', 'phone', 'printer', 'speech', 'switch']
nl = np.array(label)
for i in range(16):
    plt.figure()
    plt.imshow(X[i].reshape(256, 200).cpu(), aspect='auto')
    plt.title('_'.join(list(nl[Y[i].cpu().numpy()==1])))
    # for ii in range(3):8
    #     plt.subplot(1,3,ii+1)
    #     plt.imshow(X[i].reshape(256, 200).cpu(), aspect='auto')
    #     if ii==1 : plt.title(label[i])
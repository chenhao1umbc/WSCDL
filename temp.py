from utils2 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT(C=16, K0=1, K=3)
opts.init, opts.shuffle, opts.show_details = 'good', False, False
opts.Dh, opts.Dw, opts.batch_size = 256, 15, 3
opts.lamb, opts.eta, opts.mu = 1, 0.1, 0.01 # for sparsity penalty, label penalty, low rank penalty

# training section
x, y = load_data(opts, data='test')

label = ['alert', 'clearthroat', 'cough', 'doorknock', 'doorslam', 'drawer', 'keyboard', 'keys', 'laughter',
 'mouse', 'pageturn', 'pendrop', 'phone', 'printer', 'speech', 'switch']
nl = np.array(label)

n = 0
print('_'.join(list(nl[y[n].cpu().numpy()==1])))
plt.figure()
plt.imshow(x[n].reshape(256, 200).cpu(), aspect='auto')
plt.title('_'.join(list(nl[y[n].cpu().numpy()==1])))

print('done')
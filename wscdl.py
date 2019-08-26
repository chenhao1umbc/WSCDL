"""This is the main file to run Weakly supervised supervised dictionary learning
The default data type is torch.tensor with precision float32, the code is written for single
GPU usage. As to cpu and multi-GPU there may be small modification needed
"""

from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opts = OPT()
X, Y = load_data(opts)
D, D0, S, S0, W = init(X, Y, opts)

for i in range(opts.maxiter):
    t = time.time()
    D = updateD([D, D0, S, S0], X, Y, opts)
    print('pass D, time is ', time.time() -t ); t = time.time()
    D0 = updateD0([D, D0, S, S0], X, Y, opts)
    print('pass D0, time is ', time.time() -t); t = time.time()
    S = updateS([D, D0, S, S0, W], X, Y, opts)
    print('pass S, time is ', time.time() -t); t = time.time()
    S0 = updateS0([D, D0, S, S0], X, Y, opts)
    print('pass S0, time is ', time.time() -t); t = time.time()
    W = updateW([S, W], Y, opts)
    print('pass W, time is ', time.time() -t)


for i in range(4):
    plt.figure(i)
    plt.imshow(X[i*50:i*50+50,:], aspect='auto')
    plt.title('Class ' + str(i+1) + ' training examples')
    plt.ylabel('Example index')
    plt.xlabel('Time index')
    plt.colorbar()

ft = [featurec, feature1, feature2, feature3, feature4]
for i in ft:
    plt.plot(i.numpy())
plt.xlabel('Time index')
plt.ylabel('Magnitude')
plt.title('Plot of features')
plt.legend(['feature0', 'feature1', 'feature2', 'feature3', 'feature4'])

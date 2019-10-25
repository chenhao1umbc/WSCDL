from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

# # plt the toy data
# x = torch.arange(30).float()
# featurec = torch.sin(x * 2 * np.pi / 30)  # '''The common features'''
# feature1 = torch.sin(x * 2 * np.pi / 15) + torch.sin(x * 2 * np.pi / 10)
# feature2 = torch.sin(x * 2 * np.pi / 20) + torch.cos(x * 2 * np.pi / 5) + torch.sin(x * 2 * np.pi / 8)
# feature3 = torch.zeros(30)
# feature3[np.r_[np.arange(5), np.arange(10, 15), np.arange(20, 25)]] = 1
# feature3 = feature3 + torch.sin(x * 2 * np.pi / 13)
# feature4 = torch.zeros(30)
# feature4[np.r_[np.arange(10), np.arange(20, 30)]] = 1
# feature4 = feature4 + torch.cos(x * np.pi / 6)
# opts = OPT(lamb=10)
# opts.dev = 'cpu'
# X, Y = load_toy(opts)
# for i in range(4):
#     plt.figure(i)
#     plt.imshow(X[i*50:i*50+50,:], aspect='auto')
#     plt.title('Class ' + str(i+1) + ' training examples')
#     plt.ylabel('Example index')
#     plt.xlabel('Time index')
#     plt.colorbar()
#
# ft = [featurec, feature1, feature2, feature3, feature4]
# plt.figure()
# for i in ft:
#     plt.plot(i.numpy())
# plt.xlabel('Time index')
# plt.ylabel('Magnitude')
# plt.title('Plot of features')
# plt.legend(['feature0', 'feature1', 'feature2', 'feature3', 'feature4'])
#
# for i in range(4):
#     plt.figure()
#     plt.plot(X[i*50, :].numpy(), '-x')
#     plt.title('Class ' + str(i+1) + ' training examples')
#     plt.ylabel('Magnitude')
#     plt.xlabel('Time index')

# plot the aasp data
opts = OPT(C=16, K0=3, K=3, M=80)
X, Y = load_data(opts)
x = X[0].cpu().numpy()
print('x.shape', x.shape)
plt.figure();plt.plot(x, '--x')
plt.figure();plt.plot((x.reshape(80, 50).T).flatten(), '--x')
plt.figure();plt.imshow(x.reshape(80, 50), aspect='auto')
plt.figure();plt.imshow(x.reshape(50, 80), aspect='auto')
plt.figure();plt.imshow(x.reshape(80, 50).T, aspect='auto')


# plot result over k0 and k
x = [0.8332, 0.8323, 0.8342, 0.8338, 0.8342, 0.8325]
plt.figure();plt.plot(x, '--x')
plt.grid()
plt.title('Test accurate vs atoms in common dictionary')
plt.xlabel('Number of atoms in common dictionary')
plt.ylabel('Averaged test accurate ')

x = [0.8243, 0.8325, 0.8338, 0.8307, 0.8304]
t = [1,2,3,4,5]
plt.figure();plt.plot(t,x, '--x')
plt.grid()
plt.title('Test accurate vs atoms in individual dictionary')
plt.xlabel('Number of atoms in individual dictionary')
plt.ylabel('Averaged test accurate ')

# plot error bar
plt.figure()
y = (0.8391+0.8375+0.8345+0.8391+0.8366)/5
a= [0.8068910256410257,
 0.8167735042735043,
 0.8100071225071225,
 0.8071581196581197,
 0.8076923076923077]
plt.errorbar([1], [0.834241452991453], [[0.0025106837606837518], [0.002831196581196571]], fmt='ok', lw=0.5)
plt.errorbar([ 2], [y], [[0.00285999999], [0.0017399999999999638]], fmt='.k', lw=0.5)
plt.errorbar([3], [sum(a)/5], [[sum(a)/5- min(a)], [max(a)-sum(a)/5]], fmt='xk', lw=0.5)
plt.legend(['Ours', "You's", 'Column-wise'], loc=0)
plt.ylabel('Accuracy')
plt.xlabel('Method index')
plt.title('Classification accuracy of 5 runs')
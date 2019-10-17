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

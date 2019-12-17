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

opts = OPT(C=16, K0=3, K=3, M=80*3)
opts.transpose, opts.shuffle, opts.show_details = True, False, False  # default as true
x, y = load_data(opts, data='val')
label = ['alert', 'clearthroat', 'cough', 'doorknock', 'doorslam', 'drawer', 'keyboard', 'keys', 'laughter',
         'mouse', 'pageturn', 'pendrop', 'phone', 'printer', 'speech', 'switch']
for i in range(0, 100, 2):
    plt.figure()
    plt.imshow(x[i].reshape(50,80).t().cpu(), aspect='auto')
    plt.title('_'.join(list(np.array(label)[y.cpu()[i]!=0])))



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

# sparse coding with different initialization
acc = [0.950625, 0.888125, 0.76125, 0.7714583333333334]
recall = [0.7515723270440252, 0.4381551362683438, 0.057651991614255764, 0.049266247379454925]
precision = [1.0, 0.9976133651551312, 0.18211920529801323, 0.19831223628691982]
plt.figure()
plt.bar(list(range(4)), acc, align='edge')
plt.bar(list(range(4)),recall, width=0.6, align='edge')
plt.bar(list(range(4)), precision, width=0.4, align='edge')
plt.legend(['Accuracy', 'Recall', 'Precision'])
plt.grid()
plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['Trianing', 'Init. using trainging', 'Init. using 0', 'Init using random'])
plt.xlabel('Initialization method')
plt.ylabel('Performance')
plt.title('Comparing with different initialization methods')


# Yoo's vs ours
# K = K0 = 3, lamb = [0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]
# lamb2 = 0.01
# C=16, K0=3, K=3, M=80*3, concatanate on frequency
acc3 = [0.791875, 0.75375, 0.711875, 0.715625,  0.73625, 0.775,  0.79625, 0.77875, 0.785625, 0.794375,  0.796875, 0.803125]
recall3 = [0.0,  0.0480, 0.069, 0.075, 0.1081, 0.033, 0.039, 0.063, 0.054, 0.02102,  0.027, 0.0063]
precision3 = [0.0, 0.172, 0.13218, 0.1453, 0.2236, 0.224, 0.684, 0.333, 0.3913, 0.7,  0.9, 0.6667]

acc5 = [0.754375, 0.7675, 0.729375, 0.7475,  0.75625, 0.794375,  0.805625, 0.8025, 0.80375, 0.800625, 0.8, 0.803]
recall5 = [0.07096,  0.035, 0.1516, 0.1193, 0.1225, 0.0774, 0.08387, 0.01612, 0.0258, 0.0129, 0.0436, 0.0032]
precision5 = [0.1732, 0.1309, 0.2165, 0.22023,  0.2435, 0.3582, 0.4905, 0.3125, 0.4, 0.2352, 0.5185, 0.166]

# yoo's 1e-4, 1e-3, 0.01, 0.1
accy5= [0.8131, 0.8094, 0.8087,0.8019]
recally5 = [0.7647, 0.7, 0, 0]
precisiony5 = [0.0820, 0.0227, 0, 0]

accy3= [0.8081, 0.8056, 0.8037]  #1e-3, 0.01, 0.1
recally3 = [1, 0, 0]
precisiony3 = [0.0129, 0, 0]

plt.figure()
plt.bar(list(range(12)), acc3, align='edge')
plt.bar(list(range(12)),recall3, width=0.6, align='edge')
plt.bar(list(range(12)), precision3, width=0.4, align='edge')
plt.legend(['Accuracy', 'Recall', 'Precision'])

plt.figure()
plt.bar(list(range(12)), acc5, align='edge')
plt.bar(list(range(12)),recall5, width=0.6, align='edge')
plt.bar(list(range(12)), precision5, width=0.4, align='edge')
plt.legend(['Accuracy', 'Recall', 'Precision'])

plt.figure()
plt.bar(list(range(4)), accy5, align='edge')
plt.bar(list(range(4)),recally5, width=0.6, align='edge')
plt.bar(list(range(4)), precisiony5, width=0.4, align='edge')
plt.legend(['Accuracy', 'Recall', 'Precision'])

plt.figure()
plt.bar(list(range(3)), accy3, align='edge')
plt.bar(list(range(3)),recally3, width=0.6, align='edge')
plt.bar(list(range(3)), precisiony3, width=0.4, align='edge')
plt.legend(['Accuracy', 'Recall', 'Precision'])

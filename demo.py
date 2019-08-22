"""demo"""

from utils import *
opts = OPT()

a = torch.rand(300, 5, 500) # S0
b = torch.rand(5, 40)  # D0
k0, m = b.shape


if torch.cuda.is_available():
    a = a.cuda()  # S0
    b = b.cuda()  # D0
    k0, m = b.shape
    print('')
    print('GPU version')
    t = time.time()
    s0 = a.unsqueeze(1).unsqueeze(1)  # expand dimension for conv1d
    d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    aa, ba = a.cpu().numpy(), b.cpu().numpy()
    for i in range(300):
        for ii in range(k0):
            r1 = F.conv1d(s0[i,:, :, ii, :], d0[:, :, ii, :])
    print('Double for-loop time is :', time.time() - t)

    # t = time.time()
    # s0 = a.unsqueeze(1).unsqueeze(1)  # expand dimension for conv1d
    # d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    # for i in range(300):
    #     for ii in range(k0):
    #         r1 = np.fft.ifft(np.fft.fft(aa[i, ii, :]) * np.fft.fft(ba[ii, :], 500))
    # print('Double loop fft time is :', time.time() - t)

    t = time.time()
    s0 = a.unsqueeze(1)  # expand dimension for conv1d
    d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    for i in range(k0):
        r2 = F.conv1d(s0[:, :, i, :], d0[:, :, i, :])
    print('Single for-loop time is :', time.time() - t)

    'compare wth'
    t = time.time()
    r2 = F.conv1d(a, b.flip(1).unsqueeze(1), groups=k0)
    print('Without for-loop time is:', time.time() - t)

else:
    print('\nCPU version')
    t = time.time()
    s0 = a.unsqueeze(1).unsqueeze(1)  # expand dimension for conv1d
    d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    aa, ba = a.cpu().numpy(), b.cpu().numpy()
    for i in range(300):
        for ii in range(k0):
            r1 = F.conv1d(s0[i, :, :, ii, :], d0[:, :, ii, :])
    print('Double for-loop time is :', time.time() - t)

    # t = time.time()
    # s0 = a.unsqueeze(1).unsqueeze(1)  # expand dimension for conv1d
    # d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    # for i in range(300):
    #     for ii in range(k0):
    #         r1 = np.fft.ifft(np.fft.fft(aa[i, ii, :]) * np.fft.fft(ba[ii, :], 500))
    # print('Double loop fft time is :', time.time() - t)

    t = time.time()
    s0 = a.unsqueeze(1)  # expand dimension for conv1d
    d0 = b.flip(1).unsqueeze(0).unsqueeze(0)  # expand dimension for conv1d
    for i in range(k0):
        r2 = F.conv1d(s0[:, :, i, :], d0[:, :, i, :])
    print('Single for-loop time is :', time.time() - t)

    'compare wth'
    t = time.time()
    r2 = F.conv1d(a, b.flip(1).unsqueeze(1), groups=k0)
    print('Without for-loop time is:', time.time() - t)
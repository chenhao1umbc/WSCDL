%%%%%%setting model tunning parameters%%%%%%
winsize=5;
lamb=1e-4;
N=10;%sparsity constraints;
snr=10000;
%%%%setting parameters%%%%%%%
option.method='batch';%'online';
option.priorType='conv';%'times';
option.estepType1='chain';%'tree' for e-step;
option.addone=1;%add bias term;
option.conv=1;%or'fft' for convolution method;
option.display=0;%display the words and probablities; 
%%%%%%load data%%%
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
X = permute(rs, [2,3,1]);
x = reshape(X, 256*200, 400);
x = x./sqrt(sum(x.*x, 1));
X = reshape(x, 256, 200, 400);
Y = labels;


data.transform=0;%you can change to 1 if you want to train on synthetic spectrograms
%%%%%loading data%%%%%%%
[F,T,No_spect]=size(X);
if data.transform
    %%%%%transfer data into spectrograms%%%%%%%
    X1=X;
    X=[];
    swin=32;
    for n=1:No_spect
        s=spectrogram([sigma*randn(swin/2-1,1);X1(1,:,n)';sigma*randn(swin/2,1)],swin,swin-1,2*swin);
        X(:,:,n)=abs(s);
    end
end
%%%%%%%%%setting parameters%%%%%%%%%%
F=size(X,1);
C=size(Y,2);
K=1;
gamma=0;
runs=5;
perc=0.75;
if strcmp(option.method,'online')
    EMiterations=100000;
else
    EMiterations=1000;
end
Miterations=1;
no_train=ceil(perc*No_spect);
no_test=No_spect-no_train;
%%%%%%initialization%%%%%%
if option.addone
    no_para=F*winsize+1;
else
    no_para=F*winsize;
end


fprintf('Start trainig model\n');






clear;
close all;

%%%%%%setting model tunning parameters%%%%%%
winsize=7;
lamb=1e-4;
N=20;%sparsity constraints;
snr=10;
%%%%setting parameters%%%%%%%
option.method='batch';%'online';
option.priorType='conv';%'times';
option.estepType1='chain';%'tree' for e-step;
option.addone=1;%add bias term;
option.conv=1;%or'fft' for convolution method;
option.display=1;%display the words and probablities; 
%%%%%%load data%%%
datastr='synspect_2cluster';
load([datastr,'.mat']);
data.transform=0;%you can change to 1 if you want to train on synthetic spectrograms
%%%%%loading data%%%%%%%
[F,T,No_spect]=size(X);
if ~isempty(strfind(datastr,'sig'))
    data.addnoise=1;
else
    data.addnoise=0;
end
if data.addnoise
    %%%%%adding noise%%%%%%%%
    sigma = sqrt(avg_sigeng/(T*10^(snr/10)));
    X = X + sigma * randn(size(X));
end
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
perc=0.8;
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
for i=1:runs
    permidx(i,:)=randperm(No_spect);
    trainY = Y(permidx(i,1:no_train),:);
    trainX = X(:,:,permidx(i,1:no_train));
    trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
    wini=1e-3*randn(no_para,C,K);
    if strcmp(option.method,'batch')
        option.dsiter=100;% display for every 1000 iteration;
        tic;
        [ w{i},~,rllharr{i},garr{i}] = EMPosteriorRegularized_batch(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
         runtime_demo(i)=toc;
    else
        option.dsiter=100*No_spect;% display for every 1000 iteration;
        tic;
        [ w{i},~,rllharr{i},garr{i}] = EMPosteriorRegularized_online(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
         runtime_demo(i)=toc;
    end  
     i
    save(['synspect_2cluster_result_',option.method,'_win_',num2str(winsize),'_lamb_',num2str(lamb),'_N_',num2str(N),'_K_',num2str(K),'.mat'],'w','rllharr','permidx','option','runtime_demo');
end





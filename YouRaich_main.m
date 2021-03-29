% This is a comstomized version for performance comparison
% modified with main_demo.m
clear;
clc
close all;
addpath(genpath('You_raich'))
rng(0)

%% run toy data  --tested it works fine
% run setting_toy
% acc = zeros(5, 1); 
% for i=1:runs
%     permidx(i,:)=randperm(No_spect);
%     trainY = Y(permidx(i,1:no_train),:);
%     trainX = X(:,:,permidx(i,1:no_train));
%     trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
%     wini=1e-3*randn(no_para,C,K);
%     
%     option.dsiter=100;% display for every 1000 iteration;
%     [ w{i},~,rllharr{i},garr{i}] = EMPosteriorRegularized_batch(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
%     
%     % this part is newly added to see the signal level accuracy result
%     valX = X(:,:,permidx(i,no_train+1:end));
%     valY = Y(permidx(i,no_train+1:end),:);
%     wtx = wtimesx(w{i},valX,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
%     [y_hat, p] = get_signal_label(w{i}, valX, option);  % newly written function get the predicted signal labels
%     acc(i) = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
% end

%% train and validation ESC 10
%%%%%%setting model tunning parameters%%%%%%
winsize=5;
lamb=1e-4;
N=10;%sparsity constraints;
snr=10000;
%%%%setting parameters%%%%%%%
opt.method='batch';%'online';
opt.priorType='conv';%'times';
opt.estepType1='chain';%'tree' for e-step;
opt.addone=1;%add bias term;
opt.conv=1;%or'fft' for convolution method;
opt.display=0;%display the words and probablities; 
opt.dsiter= 10; % display for every 1000 iteration;
%%%%%%load data%%%
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
X = permute(X, [2,3,1]);

%%%%%loading data%%%%%%%
[F,T,No_spect]=size(X);
C=size(Y,2);
K=2;
gamma=0;
runs=1;
perc=0.75;  % perc * n_samples for training, default as 0.75
EMiterations = 1; 
Miterations=1;
no_train=ceil(perc*No_spect);
no_test=No_spect-no_train;

opt.C = C;
opt.F = F;
opt.T = T;
opt.K = K;
myfunc = py.importlib.import_module('myconv');
py.importlib.reload(myfunc);
opt.myfunc = myfunc; % added the myfunc to option 


%% rough tunning
lamb_pool=[10, 1, 0.1, 0.01, 1e-3, 1e-4];
winsize_pool=[30, 50, 100, 150];
N_pool=[5, 10, 20, 50, 100, 200];%sparsity constraints;

acc = zeros(runs, 1); 
rec = zeros(runs, 1); 
prec = zeros(runs, 1); 

for i=1:runs
    permidx=randperm(No_spect);
    trainY = Y(permidx(1:no_train),:);
    trainX = X(:,:,permidx(1:no_train));
    
    valX = X(:,:,permidx(no_train+1:end));
    valY = Y(permidx(no_train+1:end),:);

    x = permute(trainX, [3, 1, 2]);
    px = py.torch.tensor(py.numpy.array(x));
    px = px.cuda().float();
    opt.px = px; % added the myfunc to option
    
    for lamb = lamb_pool
        for winsize = winsize_pool
            for N = N_pool
                
opt.px = px; % opt.px will be changed in validation, do not delete               
lamb
winsize
N
opt.winsize = winsize;
trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
if opt.addone
    no_para=F*winsize+1;
else
    no_para=F*winsize;
end
wini=1e-3*randn(no_para,C,K);
[ w,~,loss,garr] = EMPosteriorRegularized_batch(...
    wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,opt,lamb);

% this part is newly added to see the signal level accuracy result
[y_hat, p] = get_signal_label(w, valX, opt);  % opt.px will be changed
acc = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
[rec, prec] = prec_rec(valY, y_hat)
f1 = 2/(1/(rec+1e-30) + 1/(prec+1e-30))

            end % end of N loop
        end % end of winsize loop
    end % end of lamb
end % end of runs

% # best lamb, winzize, N, K=1
% # 10, 30, 200
% # 10, 100, 10
% # 10, 50, 50

% # best lamb, winzize, N, k=2
% # 1, 150, 200
% # 0.01, 50, 50
% # 0.1, 100, 50

%% finer tunning
% 
% runs = 5;
% for i = 1:6
% if i == 1 [lamb, winzize, N, K] = deal(10, 30, 200, 1); end
% if i == 2 [lamb, winzize, N, K] = deal(10, 100, 10, 1); end
% if i == 3 [lamb, winzize, N, K] = deal(10, 50, 50, 1); end
% if i == 4 [lamb, winzize, N, K] = deal(1, 150, 200, 2); end
% if i == 5 [lamb, winzize, N, K] = deal(0.01, 50, 50, 2); end
% if i == 6 [lamb, winzize, N, K] = deal(0.1, 100, 50, 2); end
% acc = zeros(runs, 1); 
% f1 = zeros(runs, 1); 
% opt.C = C;
% opt.F = F;
% opt.T = T;
% opt.K = K;
% 
%     for r=1:runs
%     permidx=randperm(No_spect);
%     trainY = Y(permidx(1:no_train),:);
%     trainX = X(:,:,permidx(1:no_train));
% 
%     valX = X(:,:,permidx(no_train+1:end));
%     valY = Y(permidx(no_train+1:end),:);
% 
%     x = permute(trainX, [3, 1, 2]);
%     px = py.torch.tensor(py.numpy.array(x));
%     px = px.cuda().float();
%     opt.px = px; % added the myfunc to option
% 
%     opt.winsize = winsize;
%     opt.px = px; % opt.px will be changed in validation
%     trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
%     if opt.addone
%         no_para=F*winsize+1;
%     else
%         no_para=F*winsize;
%     end
%     wini=1e-3*randn(no_para,C,K);
%     [ w,~,loss,garr] = EMPosteriorRegularized_batch(...
%         wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,opt,lamb);
% 
%     % this part is newly added to see the signal level accuracy result
%     [y_hat, p] = get_signal_label(w, valX, opt);  % opt.px will be changed
%     acc = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
%     [rec, prec] = prec_rec(valY, y_hat)
%     f1(r) = 2/(1/(rec+1e-30) + 1/(prec+1e-30))
% 
%     end % end of runs
%     i
%     mean_f1 = mean(f1)
% end % end of i
% 
% % i = 6 is the best
% % lamb, winzize, N, K
% % 0.1, 100, 50, 2

%% test
% load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_te.mat')
% X_test = permute(X, [2,3,1]);
% Y_test = Y;
% load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
% X = permute(X, [2,3,1]);
% 
% runs = 5;
% [lamb, winzize, N, K] = deal(0.1, 100, 50, 2); 
% acc = zeros(runs, 1); 
% f1 = zeros(runs, 1); 
% opt.C = C;
% opt.F = F;
% opt.T = T;
% opt.K = K;
% 
% for r=1:runs
%     permidx=randperm(No_spect);
%     trainY = Y(permidx(1:no_train),:);
%     trainX = X(:,:,permidx(1:no_train));
% 
%     x = permute(trainX, [3, 1, 2]);
%     px = py.torch.tensor(py.numpy.array(x));
%     px = px.cuda().float();
%     opt.px = px; % added the myfunc to option
% 
%     opt.winsize = winsize;
%     opt.px = px; % opt.px will be changed in validation
%     trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
%     if opt.addone
%         no_para=F*winsize+1;
%     else
%         no_para=F*winsize;
%     end
%     wini=1e-3*randn(no_para,C,K);
%     [ w,~,loss,garr] = EMPosteriorRegularized_batch(...
%         wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,opt,lamb);
% 
%     % this part is newly added to see the signal level accuracy result
%     [y_hat, p] = get_signal_label(w, X_test, opt);  % opt.px will be changed
%     acc = sum((y_hat - Y_test) == 0, 'all')/numel(y_hat)
%     [rec, prec] = prec_rec(Y_test, y_hat)
%     f1(r) = 2/(1/(rec+1e-30) + 1/(prec+1e-30))
%     save('you_raich_yhat.mat', 'p')
% end % end of runs
% mean_f1 = mean(f1)




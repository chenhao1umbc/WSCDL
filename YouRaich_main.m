% This is a comstomized version for performance comparison
% modified with main_demo.m
clear;
close all;
addpath(genpath('You_raich'))

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
%     y_hat = get_signal_label(w{i}, valX, option);  % newly written function get the predicted signal labels
%     acc(i) = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
% end

%% train and validation ESC 10
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
option.dsiter=3;% display for every 1000 iteration;
%%%%%%load data%%%
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
X = permute(X, [2,3,1]);

%%%%%loading data%%%%%%%
[F,T,No_spect]=size(X);
C=size(Y,2);
K=1;
gamma=0;
runs=1;
perc=0.75;
EMiterations=200;
Miterations=1;
no_train=ceil(perc*No_spect);
no_test=No_spect-no_train;
%%%%%%initialization%%%%%%
if option.addone
    no_para=F*winsize+1;
else
    no_para=F*winsize;
end

%%
acc = zeros(runs, 1); 
rec = zeros(runs, 1); 
prec = zeros(runs, 1); 
l = [0.1 0.01 0.001];
for i=1:runs
    lamb = l(i);
    permidx(i,:)=randperm(No_spect);
    trainY = Y(permidx(i,1:no_train),:);
    trainX = X(:,:,permidx(i,1:no_train));
    trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
    wini=1e-3*randn(no_para,C,K);
    
    [ w{i},~,rllharr{i},garr{i}] = EMPosteriorRegularized_batch(...
        wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
    
    % this part is newly added to see the signal level accuracy result
    valX = X(:,:,permidx(i,no_train+1:end));
    valY = Y(permidx(i,no_train+1:end),:);
    wtx = wtimesx(w{i},valX,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
    y_hat = get_signal_label(w{i}, valX, option);  % newly written function get the predicted signal labels
    acc(i) = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
    [rec(i), prec(i)] = prec_rec(y_hat, valY)
end

%% test
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
X_test = permute(X, [2,3,1]);
Y_test = Y;
wtx = wtimesx(w{i},X_test,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
y_hat_test = get_signal_label(w{i}, X_test, option);  % newly written function get the predicted signal labels
test_acc = sum((y_hat_test - Y_test) == 0, 'all')/numel(y_hat_test)
[test_recall, test_prec] = prec_rec(y_hat_test, Y_test)


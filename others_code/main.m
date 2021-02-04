% This is a comstomized version for performance comparison
% modified with main_demo.m
clear;
close all;
addpath('WSCADL-YOU_master')

%% run toy data  --tested it works fine
% run setting
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

%% run aasp
run settings_aasp
runs = 3;
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
    
    option.dsiter=100;% display for every 1000 iteration;
    [ w{i},~,rllharr{i},garr{i}] = EMPosteriorRegularized_batch(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
    
    % this part is newly added to see the signal level accuracy result
    valX = X(:,:,permidx(i,no_train+1:end));
    valY = Y(permidx(i,no_train+1:end),:);
    wtx = wtimesx(w{i},valX,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
    y_hat = get_signal_label(w{i}, valX, option);  % newly written function get the predicted signal labels
    acc(i) = sum((y_hat - valY) == 0, 'all')/numel(y_hat)
    [rec(i), prec(i)] = prec_rec(y_hat, valY)

    % this part is for the test data  
    rt = '/home/chenhao1/Hpython/data/';
%     rt = '/extra/chenhao1/Hpython/data';
    load([rt,'test_256by200.mat'])
    X = permute(rs, [2,3,1]);
    x = reshape(X, 256*200, 702);
    x = x./sqrt(sum(x.*x, 1));
    X_test = reshape(x, 256, 200, 702);
    Y_test = labels;
    wtx = wtimesx(w{i},X_test,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
    y_hat_test = get_signal_label(w{i}, X_test, option);  % newly written function get the predicted signal labels
    test_acc = sum((y_hat_test - Y_test) == 0, 'all')/numel(y_hat_test)
    [test_recall, test_prec] = prec_rec(y_hat_test, Y_test)
end


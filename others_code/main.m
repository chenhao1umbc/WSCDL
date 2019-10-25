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
acc = zeros(5, 1); 
for i=1:runs
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

    x = importdata('x_test_80_50.txt');
    x = x./sqrt(sum(x.*x, 2));  % normalize data
    X_test = zeros(80, 50, size(x, 1));
    for ii = 1:size(x, 1)
        X_test(:,:, ii) = reshape(x(ii,:), 50, 80)';
    end
    Y_test = importdata('y_test_80_50.txt');
    wtx = wtimesx(w{i},X_test,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
    y_hat_test = get_signal_label(w{i}, X_test, option);  % newly written function get the predicted signal labels
    test_acc = sum((y_hat_test - Y_test) == 0, 'all')/numel(y_hat_test)

end

% [~, ind] = max(acc);  % find the best W
% x = importdata('x_test_80_50.txt');
% x = x./sqrt(sum(x.*x, 2));  % normalize data
% X_test = zeros(80, 50, size(x, 1));
% for i = 1:size(x, 1)
%     X_test(:,:, i) = reshape(x(i,:), 50, 80)';
% end
% Y_test = importdata('y_test_80_50.txt');
% wtx = wtimesx(w{ind},X_test,option);  % this function was originally defined in EMPosteriorRegularized_batch.m file
% y_hat_test = get_signal_label(w{ind}, X_test, option);  % newly written function get the predicted signal labels
% test_acc = sum((y_hat_test - Y_test) == 0, 'all')/numel(y_hat_test)

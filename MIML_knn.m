clear
clc

% add file path
addpath('./MLkNN')

% load training data
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
tr_data = zeros(800,50000);
for i= 1:800
    tr_data(i, :) = squeeze(X(i, :));
end
Y(Y==0) = -1;
tr_label = Y';

% laod testing data
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_te.mat')
te_data = zeros(200,50000);
for i= 1:200
    tr_data(i, :) = squeeze(X(i, :));
end
Y(Y==0) = -1;
te_label = Y';

%% run MIML knn 
k = 7;
smooth = 0;
% train
[Prior,PriorN,Cond,CondN]=MLKNN_train(tr_data,tr_label, k, smooth);
% test
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]= ...
    MLKNN_test(tr_data,tr_label,te_data,te_label,k,Prior,PriorN,Cond,CondN);

Average_Precision
clear
clc

% add file path
addpath('./MLkNN')

% load training data
load('/home/chenhao1/Hpython/data/aasp/train_128_880.mat')
tr_data = reshape(data, [128*128, 880]);
tr_data = tr_data';
label(label==0) = -1;
tr_label = label';  %

% laod testing data
load('/home/chenhao1/Hpython/data/aasp/test_128_222.mat')
te_data = reshape(data, [128*128, 222]);
te_data = te_data';
label(label==0) = -1;
te_label = label';

%% run MIML knn 
k = 7;
smooth = 0;
% train
[Prior,PriorN,Cond,CondN]=MLKNN_train(tr_data,tr_label, k, smooth);
% test
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]= ...
    MLKNN_test(tr_data,tr_label,te_data,te_label,k,Prior,PriorN,Cond,CondN);

Average_Precision
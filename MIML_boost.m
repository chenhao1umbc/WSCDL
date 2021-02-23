clear
clc

% add file path
addpath(genpath('./MIMLBoost_MIMLSVM'))

% load training data
load('/home/chenhao1/Hpython/data/aasp/train_128_880.mat')
tr_data = reshape(data, [128*128, 880]);
temp = tr_data';  % shape of [n_sample, n_dimension]
tr_data = cell(880,1);
for i= 1:880
    tr_data{i} = temp(i, :);
end
label(label==0) = -1;
tr_label = label;

% laod testing data
load('/home/chenhao1/Hpython/data/aasp/test_128_222.mat')
te_data = reshape(data, [128*128, 222]);
temp = te_data';  % shape of [n_sample, n_dimension]
te_data = cell(222,1);
for i= 1:222
    te_data{i} = temp(i, :);
end
label(label==0) = -1;
te_label = label;

%% run MIML boost 
% for details please refer ./MIMLBoost_MIMLSVM/sample.m

svm.type='RBF';
svm.para=1;%the value of "gamma"
cost=1;% the value of "C"
rounds = 5;

%call MIMLBoost_train and MIMLBoost_test

[classifiers,c_values,Iter]=MIMLBoost_train(tr_data,tr_label,rounds,svm,cost);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MIMLBoost_test(te_data,te_label,classifiers,c_values,Iter);
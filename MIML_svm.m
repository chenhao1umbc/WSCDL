clear
clc

% add file path
addpath(genpath('./MIMLBoost_MIMLSVM'))

% load training data
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_tr.mat')
tr_data = cell(800,1);
for i= 1:800
    tr_data{i} = squeeze(X_bag(i, 1:5, :));
end
Y(Y==0) = -1;
tr_label = Y';

% laod testing data
load('/home/chenhao1/Matlab/data_matlab/ESC10/esc10_te.mat')
te_data = cell(200,1);
for i= 1:200
    te_data{i} = squeeze(X_bag(i, 1:5, :));
end
Y(Y==0) = -1;
te_label = Y';

%% run MIML svm 
% for details please refer ./MIMLBoost_MIMLSVM/sample.m

ratio=0.2;%parameter "k" is set to be 20% of the number of training bags
svm.type='RBF';
svm.para=0.2;%the value of "gamma"
cost=1;% the value of "C"

%call MLMLSVM
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MIMLSVM(tr_data,tr_label,te_data,te_label,ratio,svm,cost);

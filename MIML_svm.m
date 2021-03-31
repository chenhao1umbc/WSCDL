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
% svm.type='RBF';
svm.type='Linear';
svm.para=0.2;%the value of "gamma"
cost=1;% the value of "C"

%call MLMLSVM
for r = 1:5
ind = randperm(800);
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=...
    MIMLSVM(tr_data(ind(1:600)),tr_label(:, ind(1:600)),te_data,te_label,ratio,svm,cost);

o = Outputs;
t = te_label;
t(t==-1) = 0;
[fpr, tpr,~, auc(r)] = perfcurve(logical(t(:)),o(:),'true');
Average_Precision
thr = mean(o);
o(o>=thr) = 1;
o(o<thr) = 0;
[rec(r), prec(r)] = prec_rec(t, o)  % it need you_raich
f1(r) = 2/(1/(rec(r)+1e-30) + 1/(prec(r)+1e-30))
end
mean(f1)
mean(auc)


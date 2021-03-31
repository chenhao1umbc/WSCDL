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

o = Outputs;
t = te_label;
t(t==-1) = 0;
Average_Precision
thr = mean(o);
o(o<thr) = 0;
o(o>=thr) = 1;
[rec, prec] = prec_rec(t, o) % it need you_raich
f1 = 2/(1/(rec+1e-30) + 1/(prec+1e-30))
[fpr, tpr,~, auc] = perfcurve(logical(t(:)),o(:),'true');
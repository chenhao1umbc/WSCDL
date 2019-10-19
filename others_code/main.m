% This is a comstomized version for performance comparison
% modified with main_demo.m
clear;
close all;
addpath('WSCADL-YOU_master')
run settings

for i=1:runs
    permidx(i,:)=randperm(No_spect);
    trainY = Y(permidx(i,1:no_train),:);
    trainX = X(:,:,permidx(i,1:no_train));
    trainNvec = N*ones(1,no_train);%N_vec(permidx(i,1:no_train));
    wini=1e-3*randn(no_para,C,K);
    if strcmp(option.method,'batch')
        option.dsiter=100;% display for every 1000 iteration;
        tic;
        [ w{i},post{i},rllharr{i},garr{i}] = EMPosteriorRegularized_batch(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
         runtime_demo(i)=toc;
    else
        option.dsiter=100*No_spect;% display for every 1000 iteration;
        tic;
        [ w{i},post{i},rllharr{i},garr{i}] = EMPosteriorRegularized_online(wini,trainX,trainY,trainNvec,EMiterations,Miterations,0,gamma,option,lamb);
         runtime_demo(i)=toc;
    end  
     i
%     save(['synspect_2cluster_result_',option.method,'_win_',num2str(winsize),'_lamb_',num2str(lamb),'_N_',num2str(N),'_K_',num2str(K),'.mat'],'w','rllharr','permidx','option','runtime_demo');
end







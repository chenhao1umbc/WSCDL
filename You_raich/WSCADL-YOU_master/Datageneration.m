clear
close all
warning off
%%%%parameters%%%%%%%
F=10;width=7;
parameters.F=F;
parameters.width = width;
xpara{1,1} = [1 5 5 1]; %%box parameters-cluster 1
ypara{1,1} = [2 2 8 8];
xpara{1,2} = [1 5 5 1]; %%box parameters-cluster 2
ypara{1,2} = [4 4 6 6];
% xpara{1,3} = [1 3 3 1]; %%box parameters-cluster3
% ypara{1,3} = [1 1 9 9];
xpara{2,1} = [1 5 2]; %%Triangle parameters-cluster 1
ypara{2,1} = [1 1 4];
xpara{2,2} = [1 7 3]; %%Triangle parameters-cluster 2
ypara{2,2} = [4 4 2];
% xpara{2,3} = [1 5 2]; %%Triangle parameters-cluster 3
% ypara{2,3} = [4 4 1];
xpara{3,1} = [0 5 5]; %%rectangle parameters-cluster 1
ypara{3,1} = [8 8 4];
xpara{3,2} = [0 5 5]; %%rectangle parameters-cluster 2
ypara{3,2} = [4 4 8];
% xpara{3,3} = [0 3 3]; %%rectangle parameters-cluster 3
% ypara{3,3} = [6 6 4];
parameters.pixel_val=[50 100 200];

% spect_vec=[50,100,150,200,1000];
% T_vec=[20,50,80,100];
spect_vec=[200];
T_vec=[50];
runs=10;
C=size(xpara,1);
K=size(xpara,2);
for No_spect=spect_vec
    permidx=zeros(runs,No_spect);
    for T=T_vec
        parameters.no_spect=No_spect;
        parameters.T=T;
        %%%%%%generate synthetic data%%%
        [ X,X2,Y,y ] = GenSynSpect( xpara,ypara,parameters );
        %%%%%%%data preprocessing %%%%%%%
        option_pre = 'AddOne';
        X2 = DataPreprocessing( X2,option_pre );
% %         for i=1:runs
% %             permidx(i,:)=randperm(No_spect);
% %                for c=1:C+1
% %                     Q = 0.05*orth(randn(size(X2{1},1),K));
% %                     for k=1:K
% %                        wini_cell{i}(:,c,k)=Q(:,k);
% %                     end
% %                 end
% % %             wini_cell{i}=0.01*randn(size(X2{1},1),C,K);%%%random initialization
% %         end
        save('synspect_2cluster.mat','X','Y','y');
    end
end

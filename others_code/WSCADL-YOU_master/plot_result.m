clear 
close all
%%%%%%%%%%%loading data%%%%%%%%%%%
load('synspect_2cluster.mat');
files=dir('synspect_2cluster_result_batch_*.mat');
[F,T,B]=size(X);
C=size(Y,2);
perc=0.8;
no_train=ceil(perc*B);
no_test=B-no_train;
for f=1:length(files)
    filename=files(f).name;
    idx1=strfind(filename,'_win_');
    idx2=strfind(filename,'_lamb_');
    idx3=strfind(filename,'_N_');
    idx4=strfind(filename,'_K_');
    idx5=strfind(filename,'.mat');
    win=str2double(filename(idx1+5:idx2-1))
    lamb=str2double(filename(idx2+6:idx3-1))
    Ns=str2double(filename(idx3+3:idx4-1))
    K=str2double(filename(idx4+3:idx5-1))
    
    load(filename);
    permidx1 = permidx;
    for i=1:length(w)
        trainY = Y(permidx1(i,1:no_train),:);
        trainX = X(:,:,permidx1(i,1:no_train));
        trainNvec = Ns*ones(1,no_train);%N_vec(permidx1(i,1:no_train));

        wnew=w{i};
        for aa=1:3 
            [temp,~]=PriorOneBag(wnew,trainX(:,:,aa),option);
            p{aa}=temp;%(1:end-1,:);
            wtx = WconvX(trainX(:,:,aa),wnew,option.addone,option.conv);
            len = size(wtx,1);
            for k=1:K
                wtx_k(:,:,k) = wtx(:,(k-1)*C+1:k*C);
            end
            wX_max = max(wtx(:));
            wX_subtract_max = wtx_k-wX_max;
            exp_wX=exp(wX_subtract_max);
        %     exp_wX2=exp(wtx_k);
            cprob{aa}=zeros(len,C+1,K);
            for k=1:K
                cprob{aa}(:,1:C,k)=exp_wX(:,:,k)./((exp(-wX_max)+sum(sum(exp_wX,3),2))*ones(1,C));
                cprob{aa}(:,C+1,k)=(1/K)*exp(-wX_max)./(exp(-wX_max)+sum(sum(exp_wX,3),2));
            end
        end
         figure(1)
         for aa=1:3
             if F==1
                 subplot(6,1,2*aa-1);plot(trainX(F,:,aa));
             else
                 subplot(6,1,2*aa-1);imagesc(trainX(:,:,aa));
             end
             ax=axis;
             title(num2str(trainY(aa,:)));
             ll=size(trainX,2)+win-1;
             lk=win;
             idxtmp=1:ll;
             idxtmp=idxtmp-lk/2;
             axis([min(idxtmp) max(idxtmp) ax(3:4)])
             %%%%%%%%%%%%%subplot of the probability%%%%%%%%%%%
             subplot(6,1,2*aa);imagesc(idxtmp,[],p{aa},[0 1]);colormap gray
             axis([min(idxtmp) max(idxtmp) 0.5 C+1.5])
         end
           [sW,sH]=size(cprob{1}(:,:,1));
           g=(reshape(1:(C+1)*K,C+1,K)');
           g=g(:);
           for aa=1:3
              newprobscale=zeros(sH*K,sW);
               for j=1:K
                   newprobscale((j-1)*(C+1)+1:j*(C+1),:)=cprob{aa}(:,:,j)';
               end
               figure(2)
               subplot(3,1,aa);imagesc(newprobscale(g,:));
               title('instance prob with scales');
           end
   %%%%%%%%%%display dictionary words at each iteration%%%%%%
        K=size(w{i},3);
        cnt=1;
        for c=1:C
            for k1=1:K
                figure(3)
                if F==1
                    subplot(C,K,cnt);plot(w{i}(1:end-1,c,k1));
                else
                    words=reshape(w{i}(1:end-1,c,k1),win,[]);
                    subplot(C,K,cnt);imagesc(fliplr(words'),[0 1]);colormap gray;
                end
                title(['c=',num2str(c),'k=',num2str(k1)])
                cnt = cnt + 1;
            end
        end

        figure(4)
        semilogy(rllharr{i});
        title('negative log likelihood')
       %%%%%%testing stage%%%%%
       hit_miml=0;hit_miml_union=0;
       for b=1:no_test
           testy{b}=y{permidx1(i,end-b+1)};
           testX1(:,:,b) = X(:,:,permidx1(i,end-b+1));
           testY(:,b) = Y(permidx1(i,end-b+1),:)';
           testN(b) = Ns;%N_vec(permidx1(i,end-b+1));
       end    
       [Pred_Labels_union{i},Pred_Labels{i},Scores{i},insacc(i),iAUC(i,:),bAUC(i,:)]= testBags( testX1,w{i},testY,testy,testN,option,win,cstr);
       for b=1:no_test
            if Pred_Labels{i}(:,b)==testY(:,b)
                hit_miml = hit_miml+1;
            end
            if Pred_Labels_union{i}(:,b)==testY(:,b)
                hit_miml_union = hit_miml_union+1;
            end
        end
        bagacc_miml(i)=hit_miml/no_test;
        bagacc_miml_union(i)=hit_miml_union/no_test;

        testY(testY==0)=-1;
        temp=Pred_Labels{i};
        temp(temp==0)=-1;
        Pred_Labels{i}=temp;
        HammingLoss(i)=Hamming_loss(Pred_Labels{i},testY);   
        RankingLoss(i)=Ranking_loss(Scores{i},testY);
        OneError(i)=One_error(Scores{i},testY);
        Coverage(i)=coverage(Scores{i},testY);
        Average_Precision(i)=Average_precision(Scores{i},testY);
        table(i,:)=[bagacc_miml(i),bagacc_miml_union(i),HammingLoss(i),...
            RankingLoss(i),OneError(i),Coverage(i),Average_Precision(i)];
        pause(0.1)
    end
    table
    disp(['signal label accuracy-probrule',' signal label accuracy-unionrule',...
     ' hamming loss',' rank loss', ' one error',' coverage',' average precision']);

end
 


function [pred_lab_union,pred_lab,score,iacc,iAUC,bAUC] = testBags( X,w,Y,y,Nbmax,option,winsize,cstr)
%TESTONEBAG test one bag accuracy
%   bag level prediction vs. instance level prediction
C=size(w,2);
B=size(X,3);
prior=PriorBags(w,X,option);
yindex=ones(1,size(Y,1));
ypower = PowerSet(yindex);
shift_table_add=BuildShiftTable_add(ypower);
for b=1:B
%     [score(:,b),~]=max(prior(1:end-1,:,b),[],2);
    for c=1:C
        tmp=1:(C+1);
        tmp(c)=[];
        score(c,b)=1-exp(sum(log(sum(prior(tmp,:,b),1))));
    end
    %%%%using dynamic programming to compute bag label probability%%%%
    pdynamic = dynamicBagProb_chain(Nbmax(b),prior(:,:,b),ypower,shift_table_add);
    [~,maxidx]=max(pdynamic);
    pred_lab(:,b)=ypower(maxidx,:);
end
%%%%%%%union rule for bag label%%%%%%
pred_lab_union=zeros(C,B);
for b=1:B
    for t=1:size(prior,2)
        [~,class_vec(t)]=max(prior(:,t,b));
    end
    uniqclass=unique(class_vec);
    pred_lab_union(uniqclass(uniqclass~=C+1),b)=1;
end
%%%%%%%%%instance AUCs and bag AUCs%%%%%%%%%
iacc=0;
totins=0;
awin=41;
for c=1:C
    cnt=1;
    delay=[];
%%%%%%%%%%%fixing delay%%%%%%%%
    for b=1:size(X,3)
        truelab(:,b)=[zeros(1,ceil(winsize/2)-1),y{b}(c,:),zeros(1,floor(winsize/2))]; % zero pads to make same length
        idx=find(truelab(:,b)==1);
        if ~isempty(idx)
            [~,idx2]=max(prior(c,:,b));
            [~,ind]=min(abs(idx-idx2));
            delay(cnt)=idx(ind)-idx2;
            cnt=cnt+1;
            %%%%%adding instance pred accuracy%%%%
            totins= totins + length(idx);
            [~,inslab]=max(prior(:,:,b),[],1);
            idx3=find(inslab==c);
            for l=1:length(idx)
                idx4=idx(l)-ceil(awin/2)+1:idx(l)+floor(awin/2);
                if ~isempty(intersect(idx3,idx4))
                    iacc = iacc + 1;
                end
            end  
        end  
    end
p=squeeze(prior(c,:,:));
p=circshift(p ,mode(delay));
p = p(ceil(winsize/2):end-floor(winsize/2),:);
temp=truelab(ceil(winsize/2):end-floor(winsize/2),:);
p_vec=p(:);
truelab_vec=temp(:);
[ iTPR,iFPR,iAUC(c) ] = ROCandAUC( p_vec,truelab_vec);%(p,truelab(60:end-60,:));
[ bTPR,bFPR,bAUC(c) ] = ROCandAUC(score(c,:)',Y(c,:)');
% figure(6)
% plot(iFPR,iTPR,'color',cstr(c,:))
% hold on
% title('detection instance ROC')
% figure(7)
% plot(bFPR,bTPR,'color',cstr(c,:))
% hold on
% title('detection bag ROC')
% legstr(c,:)=['c=',num2str(c)];
end
iacc=iacc/totins;
% figure(6)
% legend(legstr)
% hold off
% figure(7)
% legend(legstr)
% hold off

% figure(4)
% subplot(2,1,1);plot(X);
% subplot(2,1,2);imagesc(prior);
% title([num2str(ypower(maxidx,:)),'       ',num2str(Y)]);
% pause()
% %%%%%union rule of finding the index%%%%%%
% unqidx=unique(pidx);
% unqlogic=zeros(1,size(prior,1));
% unqlogic(unqidx)=1;
% hammingloss=Y~=unqlogic(1:end-1);
%%%%%%%%%computing the confusion matrix%%%%%%%%%%%%
% % C=size(y,1);
% % confusionMat = zeros(C); 
% % yr=zeros(1,size(y,2));
% % for j=1:size(y,2)
% %     yr(j)=find(y(:,j));
% % end
% % for i = 1:C 
% %     pred = pidx==i; 
% %     for j = 1:C   
% %         confusionMat(i,j) = sum ((yr==j).*pred); 
% %     end 
% % end

end

%%%%%Compute the power set of a label set%%%%%
function youtput=PowerSet(yindex)
positivelabelindex=find(yindex==1);
y=zeros(length(positivelabelindex),1);
str=Generate(y,2)-1;
youtput=zeros(size(str,1),size(yindex,2));
for i=1:size(str,1)
    youtput(i,positivelabelindex)=str(i,:);
end
end
 
function str=Generate(y,c)
if (length(y)==1)
    str=zeros(1,c)';
    for i=1:c
        str(i)=i;
    end
else
   half=floor(length(y)/2);
   str1=Generate(y(1:half),c);
   str2=Generate(y(half+1:length(y)),c);
   I1=zeros(size(str2,1),size(str1,1)); M1=[];
   for j=1:size(str1,1)
       I1(:,j)=1;
       M1=cat(1,M1,I1);
       I1(:,j)=0;
   end
   Mstr1=M1*str1;
   I2=eye(size(str2,1)); M2=[];
   for i=1:size(str1,1);
       M2=cat(1,M2,I2);
   end
   Mstr2=M2*str2;
   str=cat(2,Mstr1,Mstr2);
end
end

%%%%Genrate prior probability%%%%  
function prior=PriorBags(W,X,option)
if strcmp(option.priorType,'conv')
%     wtx = wTx(W,X,option.addone);
    wtx = WconvX(X,W,option.addone,option.conv);
else
    wtx = wtimesx(W,X,option.addone);
end
C=size(W,2);
K=size(W,3);
for b=1:size(wtx,3)
    wx_in_k=zeros(size(wtx(:,:,b),1),C,K);
    for k=1:K
        wx_in_k(:,:,k)=wtx(:,(k-1)*C+1:k*C,b);
    end 
    len=size(wx_in_k,1);
    max_out=max(max(wx_in_k,[],3),[],2);
    wx_out_k=bsxfun(@minus, wx_in_k, max_out);
    pro=sum(exp(wx_out_k),3);
    sumpro=sum(pro,2);
    p=zeros(len,C+1);
    p(:,1:C)=pro./((exp(-max_out)+sumpro)*ones(1,C));
    p(:,C+1)=1-sum(p,2);
    p=p';
    prior(:,:,b)=p;
end
end

function wx_in_k = wTx(W,X,option)%%%1-D signal
T=size(X,2);
C=size(W,2);
B=size(X,3);
if option
    width = size(W,1)-1;
else
    width = size(W,1);
end
wx_in_k=zeros(T+width-1,C,B);
for b=1:B
    for c=1:C
        if option
            wx_in_k(:,c,b) = W(end,c)*ones(T+width-1,1)+conv(X(1,:,b)',W(1:end-1,c)); %conv(X,flipud(W(1:end-1,c,k)),'same')
        else
            wx_in_k(:,c,b) = conv(X(1,:,b)',W(:,c));%conv(X,flipud(W(:,c,k)),'same')
        end
        
    end
end

end

function wx_in_k = wtimesx(W,X,option)
if option.addone
    width=size(W,1)-1;
else
    width=size(W,1);
end
B=size(X,3);
T=size(X,2);
C=size(W,2);
% wx_in_k = zeros(T,C,B);
% X_new=zeros(width,T);
% X2=[zeros(ceil(width/2)-1,1) ;X; zeros(floor(width/2),1)];
wx_in_k = zeros(T+width-1,C,B);
for b=1:B
    X_new=zeros(width,T+width-1);
    X2=[zeros(width-1,1) ;X(1,:,b)'; zeros(width-1,1)];
    for i=1:T+width-1
        X_new(:,i)=flipud(X2(i:i+width-1,1));
    end
    for c=1:C
        if option.addone
            wx_in_k(:,c,b) = W(end,c)*ones(T+width-1,1)+X_new'*W(1:end-1,c);
        else
            wx_in_k(:,c,b) = X_new'*W(:,c);
        end      
    end
end

end


%%%%using dynamic programming to compute bag label probability%%%%
function [pdynamic,hammingloss] = dynamicBagProb_chain_matlab(yindex,priorp,Y,Nbmax)
    ypower = PowerSet(yindex);
    shift_table_add=BuildShiftTable_add(ypower);
    [C,no_ins]=size(priorp);
%     priorp=PriorOneBag(w,X,option);
    pslen=size(ypower,1);
   %%%%%initialize the state probability%%%%%
    S{1}=zeros(pslen,2);
    S{1}(1,1)=priorp(C,1);
    for c=find(yindex)
        bilab=zeros(1,length(yindex));
        bilab(c)=1;
        idx=ismemberc(ypower,bilab);            
        S{1}(idx,2)=priorp(c,1);
    end
    S{1}=S{1}/sum(sum(S{1}));
    %%%%%%chain model for forward algorithm%%%%%
   for i=2:no_ins
       if i>Nbmax
           size1=Nbmax+1;
       else
           size1=size(S{i-1},2)+1;
       end
       S{i}=zeros(pslen,size1);
       for j=1:size(ypower,1)
           if i<=Nbmax
                S{i}(j,1:end-1)=S{i}(j,1:end-1)+S{i-1}(j,:)*priorp(C,i);
           else
                S{i}(j,:)=S{i}(j,:)+S{i-1}(j,:)*priorp(C,i);
           end
           for c=find(yindex)
               fidx=shift_table_add(j,c);
               if i<=Nbmax
                    S{i}(fidx,2:end)=S{i}(fidx,2:end)+S{i-1}(j,:)*priorp(c,i);
               else
                    S{i}(fidx,2:end)=S{i}(fidx,2:end)+S{i-1}(j,1:end-1)*priorp(c,i);
               end             
           end
       end
       S{i}=S{i}/sum(sum(S{i}));
   end
   pdynamic = sum(S{end},2);
   [~,maxidx]=max(pdynamic);
   hammingloss=Y~=ypower(maxidx,:);

end

function shift_table=BuildShiftTable_add(str)
shift_table=zeros(size(str));
[ps,C]=size(str);
for i=1:ps
    for j=1:C
        temp1=str(i,:);
        temp1(j)=1;
        union=temp1;
        fidx=ismemberc(str,union);
        shift_table(i,j)=fidx;
    end
end

end
 

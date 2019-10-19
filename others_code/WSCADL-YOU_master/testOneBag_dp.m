function [pred_lab,score] = testOneBag_dp( X,w,Y,Nbmax,option)
%TESTONEBAG test one bag accuracy
%   bag level prediction vs. instance level prediction
prior=PriorOneBag(w,X,option);
[score,~]=max(prior,[],2);
score=score(1:end-1);
%%%%using dynamic programming to compute bag label probability%%%%
yindex=ones(1,length(Y));
ypower = PowerSet(yindex);
shift_table_add=BuildShiftTable_add(ypower);
pdynamic = dynamicBagProb_chain(Nbmax,prior,ypower,shift_table_add);
[~,maxidx]=max(pdynamic);
pred_lab=ypower(maxidx,:);
% %%%%%union rule of finding the index%%%%%%
% [~,pidx]=max(prior);
% unqidx=unique(pidx);
% unqlogic=zeros(1,size(prior,1));
% unqlogic(unqidx)=1;
% pred_lab=unqlogic(1:end-1);
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
function p=PriorOneBag(W,X,option)
if strcmp(option.priorType,'conv')
%     wx_in_k = wtx(W,X,option.addone);
    wx_in_k = WconvX(X,W,option.addone,option.conv);
else
%     wx_in_k = wtimesx(W,X,option.addone);
    wx_in_k = X'*W;
end
C=size(W,2);
max_out=max(max(wx_in_k,[],3),[],2);
wx_out_k=bsxfun(@minus, wx_in_k, max_out);
pro=sum(exp(wx_out_k),3);
sumpro=sum(pro,2);
p=pro./(sumpro*ones(1,C));
p=p';
end

function wx_in_k = wtx(W,X,option)
F=size(X,1);
T=size(X,2);
K=size(W,3);
C=size(W,2);
if strcmp(option,'AddOne')
    width=(size(w,1)-1)/F;
else
    width=size(w,1)/F;
end
wx_in_k = zeros(T,C,K);
for f=1:F
    for c=1:C
        for k=1:K
            if strcmp(option,'AddOne')
                wx_in_k(:,c,k) = wx_in_k(:,c,k) + W(end,c,k)*ones(T,1)/F+conv(X(f,:)',flipud(W((f-1)*width+1:f*width,c,k)),'same');
            else
                wx_in_k(:,c,k) = wx_in_k(:,c,k) + conv(X(f,:)',flipud(W((f-1)*width+1:f*width,c,k)),'same');
            end

        end
    end
end

end

function wx_in_k = wtimesx(W,X,option)
F=size(X,1);
T=size(X,2);
K=size(W,3);
C=size(W,2);
if strcmp(option,'AddOne')
    width=(size(w,1)-1)/F;
else
    width=size(w,1)/F;
end
wx_in_k = zeros(T,C,K);
for f=1:F
    X_new=zeros(width,T);
    X2=[zeros(ceil(width/2)-1,1) ;X(f,:)'; zeros(floor(width/2),1)];
    for i=1:T
        X_new(:,i)=X2(i:i+width-1);
    end
    for c=1:C
        for k=1:K
            if strcmp(option,'AddOne')
                wx_in_k(:,c,k) = wx_in_k(:,c,k) + W(end,c,k)*ones(T,1)+X_new'*W((f-1)*width+1:f*width,c,k);
            else
                wx_in_k(:,c,k) = wx_in_k(:,c,k) + X_new'*W((f-1)*width+1:f*width,c,k);
            end      
        end
    end
end

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

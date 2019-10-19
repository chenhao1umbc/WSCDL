%%%%Genrate prior probability%%%%  
function [p,wx_in_k]=PriorOneBag(W,X,option)
if strcmp(option.priorType,'conv')
%     wx_in_k = wtx(W,X,option.addone);
    wx = WconvX(X,W,option.addone,option.conv);
else
    wx = wtimesx(W,X,option.addone);
end
% C=size(W,2);
% max_out=max(max(wx_in_k,[],3),[],2);
% wx_out_k=bsxfun(@minus, wx_in_k, max_out);
% pro=sum(exp(wx_out_k),3);
% sumpro=sum(pro,2);
% p=pro./(sumpro*ones(1,C));
% p=p';
C=size(W,2);K=size(W,3);
wx_in_k=zeros(size(wx,1),C,K);
for k=1:K
    wx_in_k(:,:,k)=wx(:,(k-1)*C+1:k*C);
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
end

function wx_in_k = wtx(W,X,option)
T=size(X,1);
K=size(W,3);
C=size(W,2);
wx_in_k = zeros(T,C,K);
for c=1:C
    for k=1:K
        if strcmp(option,'AddOne')
            wx_in_k(:,c,k) = W(end,c,k)*ones(T,1)+conv(X,flipud(W(1:end-1,c,k)),'same'); 
        else
            wx_in_k(:,c,k) = conv(X,flipud(W(:,c,k)),'same');
        end
        
    end
end

end

function wx_in_k = wtimesx(W,X,option)
if strcmp(option,'AddOne')
    width=size(W,1)-1;
else
    width=size(W,1);
end
T=size(X,1);
K=size(W,3);
C=size(W,2);
wx_in_k = zeros(T,C,K);
X_new=zeros(width,T);
X2=[zeros(ceil(width/2)-1,1) ;X; zeros(floor(width/2),1)];
for i=1:T
    X_new(:,i)=X2(i:i+width-1);
end
for c=1:C
    for k=1:K
        if strcmp(option,'AddOne')
            wx_in_k(:,c,k) = W(end,c,k)*ones(T,1)+X_new'*W(:,c,k);
        else
            wx_in_k(:,c,k) = X_new'*W(:,c,k);
        end      
    end
end

end




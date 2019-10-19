function wx_in_k = wtimesx(W,X,option)
if option.addone
    width=size(W,1)-1;
else
    width=size(W,1);
end
B=size(X,3);  % batch size
T=size(X,2);  % X time lenght
C=size(W,2);  % classes
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
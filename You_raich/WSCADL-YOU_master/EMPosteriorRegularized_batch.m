function [ w,post,rllharr,garr] = EMPosteriorRegularized_batch(w,X,Y,Nvec,EMiterations,Miterations,lambda,gamma,option,lam)
%% SECTION TITLE
% DESCRIPTIVE TEXT
% alpha=1.5;
% no_ins=size(X,1)-size(w,1)+1;
% B=size(X,2);
% Prior=zeros(size(w,2),no_ins,B);
% for i=1:B
%     Prior(:,:,i)=PriorOneBag(w,X(:,i),option.priorType);
%     %%%%%adding weights on novel probability%%%%%
%     cnt=1;
%     while(sum(Prior(end,:,i)) < no_ins-N && cnt < 100)
%         w(:,end,:) = w(:,end,:)*alpha;
%         Prior(:,:,i)=PriorOneBag(w,X(:,i),option.priorType);
%         cnt=cnt+1
%     end
% end
%%%%Supervised dictionary learning algorithm using EM
curr_iter=1;
step=1;
rllharr=zeros(1,EMiterations*Miterations);
garr=zeros(1,EMiterations*Miterations);
[F,T,B]=size(X);
[~,C,K]=size(w);
win=floor(size(w,1)/F);
eps=1e-10;
tempidx=randperm(B,5);
fprintf('Start trainig model\n')

myfunc = py.importlib.import_module('myconv');
py.importlib.reload(myfunc);
option.myfunc = myfunc; % added the myfunc to option 

x = permute(X, [3, 1, 2]);
px = py.torch.tensor(py.numpy.array(x));
px = px.cuda().float();
option.px = px; % added the myfunc to option 

while(curr_iter<=EMiterations*Miterations)
    wold = w;
    if strcmp(option.priorType,'conv')
%         wxold_cell=WconvX(X,wold,option.addone,option.conv);
        wxold_cell=wconx(wold, option);
    else
        for i=1:B
             wxold_cell(:,:,i) = wtimesx(wold,X(1,:,i)',option.addone);
        end
    end
%     wx_in_k = wtx(w,X,option.addone);%%for checking the c code 
%     wx_in_k2 = wtimesx(w,X,option.addone);
    for i=1:B      
        if size(wold,3)==1
            cprob_cell{i}=ones(size(wxold_cell,1),size(wold,2),size(wold,3));
        else
            cprob_cell{i} = probcluster(wxold_cell(:,:,i),K,C);
        end
    end
    [post,pyn,Prior]=RegularizedExpectationStep(wxold_cell,Y,Nvec,option);
%     if size(wold,3)==1
%         const=0;
%     else
%         const = compConstforG(wxold_cell,cprob_cell,post);
%     end
    const=0;
    for j=1:Miterations
        if option.addone
            rllharr(curr_iter)=-sum(pyn)/B+0.5*lam*sum(sum(sum(wold(1:end-1,:).^2,3),2));
        else
            rllharr(curr_iter)=-sum(pyn)/B+0.5*lam*sum(sum(sum(wold.^2,3),2));
        end
        garr(curr_iter)=Gtilde(post,X,wold,cprob_cell,const,option,lambda,gamma,lam);
%         step=10*step;
        [w, step, ~]=MaximizationStep(X,wold,wxold_cell,cprob_cell,const,post,lambda,gamma,step,option,lam);
    end
        C=size(w,2);
        K=size(w,3);
%         if count > 2 && rllharr(count-1)-rllharr(count) <= eps
%             break;
%         end
                                 
        if (rem(curr_iter, option.dsiter)==0)
            curr_iter
            if option.display
            %%%%%%%%%%%%display dictionary words at each iteration%%%%%%
               cnt=1;
               for c=1:C
                   for k=1:K
                       figure(1)
                       if F==1
                           subplot(C,K,cnt);plot(w(1:end-1,c,k));
                       else
                           words=reshape(w(1:end-1,c,k),win,[]);
                           subplot(C,K,cnt);imagesc(words');
                       end
                       cnt=cnt+1;
                   end
               end
               figure(2)
               for aa=1:5
                   subplot(5,1,aa);imagesc(post{tempidx(aa)}');
                   title(num2str(Y(tempidx(aa),:)));
               end
               figure(3)
               for aa=1:5
                   if F==1
                       subplot(5,1,aa);plot(X(:,:,tempidx(aa)));
                   else
                       subplot(5,1,aa);imagesc(X(:,:,tempidx(aa)));
                   end
                   title(num2str(Y(tempidx(aa),:)));
               end

               figure(4)
               semilogy(rllharr);
               title('negative log likelihood');
    %              figure(5)
    %             semilogy(garr);
    %             title('negative log auxilliary');
               pause(0.1);
            end
        end
        %check convergence
        if curr_iter >3 && (abs(rllharr(curr_iter) -rllharr(curr_iter-1))/ abs(rllharr(curr_iter)) <1e-3)
            fprintf('loss change is too small and stop tranining\n')
            curr_iter
            break
        end
        curr_iter=curr_iter+1;
end

end

function [postprob,Pyn,Prior]=RegularizedExpectationStep(wX,Y,Nvec,option)
postprob={};
[no_ins,CK,B]=size(wX);
C=size(Y,2);
K=CK/C;
Pyn=zeros(B,1);
Prior=zeros(C+1,no_ins,B);
for i=1:B
    Prior(:,:,i) = PriorOneBag(wX(:,:,i),C,K);
    classes_in_bag = sum(Y(i,:));
    N = Nvec(i);
    if(classes_in_bag>0)
        [ypower,~,shift_table_add,shift_table_tree]=generatetool(Y(i,:),option.estepType1);
        if strcmp(option.estepType1,'tree')
            [postprob{i},Pyn(i,:)]=RegularizedStatePosterior_tree_dp_simplified(N, Prior(:,:,i),ypower,shift_table_tree);%,stprior,stpost
            postprob{i}=postprob{i}';
        elseif strcmp(option.estepType1,'chain')
                [postprob{i},Pyn(i,:)]=RegularizedStatePosterior_chain_dp_simplified(N, Prior(:,:,i),ypower,shift_table_add);
                postprob{i}=postprob{i}';
        else
            disp('wrong option, please enter tree or chain instead')
            exit;
        end
    else
        no_of_class=size(Y(i,:),2)+1;
        postprob{i}=zeros(no_ins,no_of_class);
        postprob{i}(:,no_of_class)=postprob{i}(:,no_of_class)+1;
        Pyn(i,1)=sum(log(Prior(end,:,i)));
    end
end

% % if strcmp(option.estepType2,'tree')
% %     [postn,~]=RegularizedPosteriorN_tree(priorn,N);
% % elseif strcmp(option.estepType2,'chain')
% %     [postn,~]=RegularizedPosteriorN_chain(priorn,N);
% % else
% %     disp('wrong option, please enter tree or chain instead')
% %     exit;
% % end

end


function [w_new, step, enough]=MaximizationStep(X,w,wX,clusterprob_cell,const,p,lambda,gamma,step_init,option,lam)

% grad=Gradient(X,w,wX,clusterprob_cell,p,lambda,gamma,option,lam);
% % norm(grad(:))
% % %%%%%check gradient%%%%
% % llh=Gtilde(p,X,w,clusterprob_cell,const,option,lambda,gamma,lam);
% % eps = 1e-5;
% % pos=randi(size(w,1),1);
% % wnew = w;
% % wnew(pos,1,1)=w(pos,1,1)+eps;
% % llhnew=Gtilde(p,X,wnew,clusterprob_cell,const,option,lambda,gamma,lam);
% % grad(pos,1,1)-(llhnew-llh)/eps
% % pause()
% [w_new,step, enough]=BackTracking1(w,X,grad,p,clusterprob_cell,const,0.5,0.7,step_init,option,lambda,gamma,lam);
options2 = struct('GradObj','on','Display','off','LargeScale','off','HessUpdate','lbfgs','GoalsExactAchieve',0,'MaxIter',1);
[w_new,step,enough,output,grad] = fminlbfgs(@(w)Gtilde(p,X,w,clusterprob_cell,const,option,lambda,gamma,lam),w,options2);

end

%%%%Compute constant part for gtilde%%%%
function const = compConstforG(wX_old_cell,clusterProb_cell,posterior_cell)
sum1=0;
sum2=0;
C=size(clusterProb_cell{1},2);
K=size(clusterProb_cell{1},3);
wX_old_temp=zeros(size(wX_old_cell,1),C,K);
for b=1:size(wX_old_cell,3)
    for k=1:K
        wX_old_temp(:,:,k)=wX_old_cell(:,(k-1)*C+1:k*C,b);
    end
    wX_max=max(wX_old_temp,[],3);
    wX_subtr_max=zeros(size(wX_old_temp));
    for k =1:K
        wX_subtr_max(:,:,k)=wX_old_temp(:,:,k)-wX_max;
    end
    sum1 = sum1 + sum(sum(posterior_cell{b}(:,1:C).*(log(sum(exp(wX_subtr_max),3))+wX_max)));
    sum2 = sum2 + sum(sum(posterior_cell{b}(:,1:C).*sum(clusterProb_cell{b}.*wX_old_temp,3)));
end
const = sum1 - sum2;
end


%%%%%Compute the gtilde function%%%%%%%
function [llh, grad]=Gtilde(p,X,w,clusterProb_cell,const,option,lambda,gamma,lam)
llh_part=0;
if strcmp(option.priorType,'conv')
%     wX=WconvX(X,w,option.addone,option.conv);
    wX=wconx(w, option);
else
     for i=1:size(X,3)
          wX(:,:,i) = wtimesx(wold,X(1,:,i)',option.addone);
     end
end
C=size(w,2);
K=size(w,3);
B=size(wX,3);
for b=1:B
    wX_b=wX(:,:,b);
    wX_max=max(wX_b,[],2);
    wX_substract=zeros(size(wX_b,1),C,K);
    wX_k=zeros(size(wX_b,1),C,K);
    for k=1:K
        wX_k(:,:,k)=wX_b(:,(k-1)*C+1:k*C);
        wX_substract(:,:,k)=wX_k(:,:,k)-wX_max*ones(1,size(w,2));
    end
    llh=sum(sum(p{b}(:,1:C).*sum(clusterProb_cell{b}.*wX_k,3),2)-log(exp(-wX_max)+sum(sum(exp(wX_substract),3),2))-wX_max);
    llh2(b,1)=llh;
    llh_part=llh_part-llh;
end
% for b=1:size(X,3)
%     [llh_part(b),wX{b}]=GtildeOneBag(p{b},X(:,:,b),w,clusterProb_cell{b},option);
% end
if option.addone
    llh=(llh_part-const)/B+0.5*lam*sum(sum(sum(w(1:end-1,:).^2,3),2));
else
    llh=(llh_part-const)/B+0.5*lam*sum(sum(sum(w.^2,3),2));
end

if(lambda~=0)
  regularization_=Regularization(A,w,lambda);
  llh=llh-regularization_;
end
if(gamma~=0)
  llh_norm=0;
  for c=1:size(w,2)
      for k=1:size(w,3)
           llh_norm=llh_norm+gamma*norm(w(:,c,k));
      end
  end
  llh=llh-llh_norm;
end

if nargout > 1
    grad=Gradient(X,w,wX,clusterProb_cell,p,lambda,gamma,option,lam);
%     grad=grad(:);
end

end
 
%%%%%Compute the gradient%%%%%%%% 
function grad=Gradient(X,w,wX,cprob,p,lambda,gamma,option,lam)
C=size(w,2);
K=size(w,3);
no_ins=size(wX,1);
B=size(X,3);
grad=zeros(size(w));
Activations = zeros(no_ins,C*K,B);
for b=1:B
    wx_in_k=wX(:,:,b);
    wxmax=max(wx_in_k,[],2);
    wx_out_k=bsxfun(@minus, wx_in_k, wxmax);
    exp_wX=exp(wx_out_k);
    sum_wX=sum(exp_wX,2);
    temp_k=zeros(size(exp_wX));
    for k=1:K
        temp_k(:,(k-1)*C+1:k*C) = p{b}(:,1:C).*cprob{b}(:,:,k);
    end
    Activations(:,:,b)=temp_k - exp_wX./((exp(-wxmax)+sum_wX)*ones(1,C*K));
end
F=size(X,1);
if (option.addone)
    width=(size(w,1)-1)/F;
else
    width=size(w,1)/F;
end
    if strcmp(option.priorType,'conv')
        %%%%%%using c for fast computation
        tempgrad = gradconv(X,Activations,width,C,option.addone,option.conv);%%conv 1, fft 0
        grad = grad + reshape(sum(tempgrad,3),[],C,K);
    else
        for b=1:B
%             spect=[zeros(ceil(width/2)-1,1) ;X(1,:,b)'; zeros(floor(width/2),1)];
            spect=[zeros(width-1,1) ;X(1,:,b)'; zeros(width-1,1)];
            X_new=zeros(width,no_ins);
            for i=1:no_ins
%                 X_new(:,i)=spect(i:i+width-1,1);
                X_new(:,i)=flipud(spect(i:i+width-1,1));
            end
            for k=1:K
                if option.addone
                    grad2(1:end-1,:,k) = grad2(1:end-1,:,k) + X_new*Activations(:,(k-1)*C+1:k*C,b);
                    grad2(end,:,k) = grad2(end,:,k) + sum(Activations(:,(k-1)*C+1:k*C,b),1);
                else
                    grad2(:,:,k) = grad2(:,:,k) + X_new*Activations(:,(k-1)*C+1:k*C,b);
                end
            end
        end
    end
grad = grad/B;
if option.addone
    grad(1:end-1,:) = lam*w(1:end-1,:) - grad(1:end-1,:);
    grad(end,:) = -grad(end,:);
else
    grad = lam*w - grad;
end

if(lambda~=0)
grad_reg=GradientOfRegularization(A,w,lambda);
grad=grad-grad_reg;
end
 
if(gamma~=0)
w_temp=w;
%%%%%Don't regularize the last feature of X (which is 1, the bias term)
w_temp(size(w_temp,1),:)=0;
grad=grad-gamma*2*w_temp;    
end
 
end

%%%%%backtracking line search%%%%%
function [w_new, step, enough]=BackTracking1(w,X,grad,p,cprob_cell,const,alpha,beta,step_init,option,lambda,gamma,lam)
stop=0;
enough=0;
f=Gtilde(p,X,w,cprob_cell,const,option,lambda,gamma,lam);
% grad2=Gradient(X,w,cprob_cell,p,lambda,gamma,option,lam);
f1=grad(:);
%[grad2(:),grad(:)]
step=step_init;
% % %%%%%%%%%%%%debugging for gradient and objective%%%%%%
% % cnt=0;
% % rng=linspace(0,step_init,50);
% % for step2=rng
% %     cnt=cnt+1;
% %     w_step2=w-step2*grad;
% %     f_w_step2(cnt)=Gtilde(p,X,w_step2,cprob_cell,const,option,lambda,gamma,lam);
% % end
% % %w2=w;
% % eps88=1e-8;
% % tmp7=randn(size(w));
% % w2=w+eps88*tmp7;
% % f2=Gtilde(p,X,w2,cprob_cell,const,option,lambda,gamma,lam);
% % ((f2-f)/eps88 - sum(sum(sum(grad.*tmp7))))/(sum(sum(sum(grad.*tmp7))))
% % figure(6)
% % plot(rng,f_w_step2)
% % title('every iteration Q');
% % pause()
% % %%%%%%%%%%%%debugging for gradient and objective%%%%%%
while(stop==0)
    w_step=w-step*grad;
    f_w_step=Gtilde(p,X,w_step,cprob_cell,const,option,lambda,gamma,lam);
    if(f_w_step<f-alpha*step*(f1'*f1))     
            stop=1;
    else
    step=step*beta;
    if ((step<1e-12)&&(f_w_step<=f))
        stop=1;
        enough=1;
    end
    end
end
w_new=w_step;
end


%%%%Compute cluster probability%%%
function cprob=probcluster(wX,K,C)
wX_k=zeros(size(wX,1),C,K);
for k=1:K
    wX_k(:,:,k)=wX(:,(k-1)*C+1:k*C);
end
wX_maxk = max(wX_k,[],3);
wX_subtract_max = zeros(size(wX_k));
cprob = zeros(size(wX_k));
for k=1:K
    wX_subtract_max(:,:,k)=wX_k(:,:,k)-wX_maxk;
end
exp_wX=exp(wX_subtract_max);
for k=1:K
    cprob(:,:,k)=exp_wX(:,:,k)./sum(exp_wX,3);
end
end


%%%%Genrate prior probability%%%%  
function p=PriorOneBag(wx,C,K)
% C=size(wx_in_k,2);
% max_out=max(max(wx_in_k,[],3),[],2);
% wx_out_k=bsxfun(@minus, wx_in_k, max_out);
% pro=sum(exp(wx_out_k),3);
% sumpro=sum(pro,2);
% p=pro./(sumpro*ones(1,C));
% p=p';
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



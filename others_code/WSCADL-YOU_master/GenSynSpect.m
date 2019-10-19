function [ X,X2,Y,y ] = GenSynSpect( xpara,ypara,parameters )
%GENSYNSPECT generate synthetic spectrograms
%   input: xpara, ypara-shape parameters; 
%   output: features and labels(SISL and MIML)
%%%%loading parameters%%%%%%%
F=parameters.F;
T=parameters.T;
width = parameters.width;
imageSize = [F T+width-1];
No_spect=parameters.no_spect;
No_instance=T;
pixel_val=parameters.pixel_val;
%%%generate multinomial probability%%%
C=size(xpara,1);
alpha=[ones(1,C) 40];
beta=1;
Y=zeros(No_spect,C);
X=zeros(F,T,No_spect);
clusterbag=randi(size(xpara,2),1,No_spect);
for i=1:No_spect
    p = gamrnd(alpha,1,1,length(alpha));
    p = p ./ repmat(sum(p,2),beta,length(alpha));
    cump = cumsum(p(:));
    r = rand(1,No_instance);   % rand numbers are in the open interval (0,1)
    [~,inst_lab] = histc(r,[0;cump/cump(end)]);
    classlab=unique(inst_lab);
    %%%%%%%%%generate noise bag%%%%
    while length(classlab)~=1 && i<= 10
        p = gamrnd(alpha,1,1,length(alpha));
        p = p ./ repmat(sum(p,2),beta,length(alpha));
        cump = cumsum(p(:));
        r = rand(1,No_instance);   % rand numbers are in the open interval (0,1)
        [~,inst_lab] = histc(r,[0;cump/cump(end)]);
        classlab=unique(inst_lab);
    end
    %%%%%%%%%generate single class bag%%%%
    while length(classlab)~=2 && (i>10 && i<= 50)
        p = gamrnd(alpha,1,1,length(alpha));
        p = p ./ repmat(sum(p,2),beta,length(alpha));
        cump = cumsum(p(:));
        r = rand(1,No_instance);   % rand numbers are in the open interval (0,1)
        [~,inst_lab] = histc(r,[0;cump/cump(end)]);
        classlab=unique(inst_lab);
    end
    %%%%%%%%%end of generating%%%%%%%%%%%%%%%
    y{i}=zeros(length(alpha),No_instance);
    for j=1:No_instance
        y{i}(inst_lab(j),j)=1;
    end
    normal_class=zeros(size(inst_lab));
    normal_class(inst_lab~=C+1)=inst_lab(inst_lab~=C+1);
    image=zeros(imageSize);
    pos=find(normal_class);
    clusterlab=clusterbag(i);
    for j=1:length(pos)
        mask = poly2mask(xpara{normal_class(pos(j)),clusterlab}+pos(j),ypara{normal_class(pos(j)),clusterlab},F,T+width-1);
%         image(mask)=image(mask)+pixel_val(normal_class(pos(j)));
        image(mask)=image(mask)+rand(1,1)*(max(pixel_val)-min(pixel_val))+min(pixel_val);
    end
    X(:,:,i)=image(:,ceil(width/2):end-floor(width/2))/mean(pixel_val)+0.05*randn(F,T);
    uni_lab=unique(inst_lab(inst_lab~=C+1));
    Y(i,uni_lab)=1;
    figure(1)
    imagesc(X(:,:,i),[0 2]);colormap gray;
    title(num2str((1:C+1)*y{i}));
    pause(0.1)
end
for i=1:No_spect
    X_new=[zeros(F,ceil(width/2)-1) ,X(:,:,i), zeros(F,floor(width/2))];
    for j=1:No_instance
        temp=X_new(:,j:j+width-1)';
        X2{i}(:,j)=temp(:);
    end
end
% W=randn(size(X2{1},1),1);
% xtw1=X2{1}'*W;
% xtw2=zeros(T,1);
% for f=1:F
%     xtw2=xtw2+conv(X{1}(f,:)',flipud(W((f-1)*width+1:f*width,1)),'same'); 
% end

end


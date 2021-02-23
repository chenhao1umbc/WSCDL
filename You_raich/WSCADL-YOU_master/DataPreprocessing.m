function X_out = DataPreprocessing( X,option )
%DATAPREPROCESSING is the pre-processing step of the input data 
%   including data normalize and add one to the feature
    switch option
        case char('AddOne')
            X_out = AddOne(X);
        case char('Normalize')
            X_out = DataNormalize(X);
        case char('ColumnNormalize')
            X_out = DataColumnNormalize(X);
        case char('Both')
            X_out = DataNormalize(X);
            X_out = AddOne(X_out);
        otherwise
            disp('please input right parameter')
            X_out=X;
    end

end


function X_out=DataNormalize(X)
 
firstmm=zeros(size(X{1},1),length(X));
secondmm=zeros(size(X{1},1),length(X));
n=0;
 
for i=1:length(X)
    if(size(X{i},2)>0)
        firstmm(:,i)=sum(X{i},2);
        secondmm(:,i)=sum(X{i}.*X{i},2);
        n=n+size(X{i},2);
    end
end
 
mean2=sum(firstmm,2)/n;
 
secondmm2=sum(secondmm,2)/n;
 
variance2=secondmm2-mean2.*mean2;
std2=sqrt(variance2*n/(n-1));
ind=find(std2==0);std2(ind)=abs(mean2(ind))+1;
 
for i=1:length(X)
    if(size(X{i},2)>0)
        for j=1:size(X{i},2)
            temp=X{i}(:,j)-mean2;
            X{i}(:,j)=temp./std2;
        end
    end
end
 
X_out=X;
end

function X=DataColumnNormalize(X)
for i=1:length(X)
    for j=1:size(X{i},2)
        X{i}(:,j)=X{i}(:,j)/norm(X{i}(:,j),2);
    end
end
end

function X_out=AddOne(X)
for i=1:length(X)
    if(size(X{i},2)>0)
        X_out{i}=[X{i}; ones(1,size(X{i},2))];
    end
end
end

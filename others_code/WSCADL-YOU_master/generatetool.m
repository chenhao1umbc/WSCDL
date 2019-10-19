function [str,shift_table,shift_table_add,shift_table_tree]=generatetool(yindex,option)
    str=PowerSet(yindex);
    shift_table=zeros(size(str,1),sum(yindex));
    shift_table_add=zeros(size(str));
    shift_table_tree=zeros(size(str,1));
    if strcmp(option,'chain')
        shift_table_add=BuildShiftTable_add(str);
    else
        if strcmp(option,'tree')
            shift_table_tree=BuildShiftTable_tree(str);
        else
            shift_table=BuildShiftTable(str);
        end
    end
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

function shift_table=BuildShiftTable_tree(str)
ps=size(str,1);
shift_table=zeros(ps,ps);
for i=1:ps
    for j=1:ps
        union= +(str(i,:) | str(j,:));
        fidx=ismemberc(str,union);
        shift_table(i,j)=fidx;
    end
end

end

function shift_table=BuildShiftTable(str)
possiblelabel=find(str(size(str,1),:)==1);
shift_table=zeros(size(str,1),length(possiblelabel));
for j=1:length(possiblelabel)
    for i=1:size(str,1)
        shift_table(i,j)=Shift(str,possiblelabel(j), str(i,:));
    end
end
end
 
function strshift=Shift(str,index, currentstr)
tempstr=(str(size(str,1),:)==1);
if(currentstr(index)==1)
currentstr(index)=0;
currentstr=currentstr(tempstr);
d=bi2de_new(fliplr(currentstr));
strshift=d+1;
else
   strshift=-1;  
end
end

function d=bi2de_new(b)
max_length = 1024;
pow2vector = 2.^(0:1:(size(b,2)-1));
size_B = min(max_length,size(b,2));
d = b(:,1:size_B)*pow2vector(:,1:size_B).';
end
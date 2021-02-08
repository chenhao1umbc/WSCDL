% this function will make the train and test shuffled, normalized
% and split with 80% for training and 20% for test
clear
clc 

% rt = '/extra/chenhao1/data_sets/AASP/'; % spss6
rt = '/home/chenhao1/Hpython/data/aasp/';
% load([rt,'train_256by200.mat'])
load([rt,'train_128.mat'])
X = permute(data, [2,3,1]);
X = flip(X, 1);
x = reshape(X, 128*128, 400);
x = x./sqrt(sum(x.*x, 1));
X = reshape(x, 128, 128, 400);
Y = labels;
x1 = X; y1 = Y;
clear X
clear Y

% rt = '/extra/chenhao1/data_sets/AASP/'; % spss6
rt = '/home/chenhao1/Hpython/data/aasp/';
% load([rt,'test_256by200.mat'])
load([rt,'test_128.mat'])
X = permute(data, [2,3,1]);
X = flip(X, 1);
x = reshape(X, 128*128, 702);
x = x./sqrt(sum(x.*x, 1));
X = reshape(x, 128, 128, 702);
Y = labels;
x2 = X; y2 = Y;

rng(1)
ind = randperm(702 + 400);
itr = ind(1:880);
ite = ind(881:end);
x = cat(3, x1, x2);
y = cat(1, y1, y2);

data = x(:,:,itr);
label = y(itr,:);
save([rt,'train_128_880.mat'], 'data', 'label')

data = x(:,:,ite);
label = y(ite,:);
save([rt,'test_128_880.mat'], 'data', 'label')

imagesc(y(itr,:))
figure;imagesc(y(ite,:))
sum(y(itr,14))

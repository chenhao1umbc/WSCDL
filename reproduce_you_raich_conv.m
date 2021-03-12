clear
clc
close all

%you_raich convolution has some randomness, I guess they for this kind of
%algorithm for speeding up, they are fastest for small data. As for the
%large data they are not fast. Also for the large data, the randomness 
% is almost gone.

% % samll setting
% F = 10;
% ws = 4;
% T = 10;
% n = 15;
% C = 2;

% large setting
F = 100;
ws = 30;
T = 500;
n = 100;
C = 2;

opt.F = F;
opt.winsize = ws;
opt.C = C;

opt.conv = 'conv';
opt.addone = 0; %1 or 0

X = rand(F,T,n);
if opt.addone
    w = rand(F*ws+1,C);
    ww = w(1:end-1, :);
    bias = w(end,:);
    ww = reshape(ww, [opt.F, opt.winsize, opt.C]);
else
    w = rand(F*ws,C);
    ww = reshape(w, [opt.F, opt.winsize, opt.C]);
end
% w = w- w;   % this is very simple to show if conv implemented correct
% w(1,1,1) =1;
% w(1,1,2) =1;

myfunc = py.importlib.import_module('myconv');
py.importlib.reload(myfunc);
opt.myfunc = myfunc; % added the myfunc to option 

x = permute(X, [3, 1, 2]);
px = py.torch.tensor(py.numpy.array(x));
px = px.cuda().float();
opt.px = px; % added the myfunc to option 


tic
a=WconvX(X,w,opt.addone,opt.conv);
toc

tic
b=wconx(w, opt);
% if opt.addone  % this part is in wconv now
%     b(:,1,:) = bias(1) + b(:,1,:);
%     b(:,2,:) = bias(2) + b(:,2,:);
% end
toc

tic
res = zeros(F, (T+ws-1), C, n);
for i = 1:F
for ii = 1:C
for iii = 1:n
res(i, :, ii, iii) = ifft(fft(ww(i,:, ii), (T+ws-1)) .*fft(X(i, :, iii), (T+ws-1)));
end
end
end
c = squeeze(sum(res, 1));
if opt.addone
    c(:,1,:) = bias(1) + c(:,1,:);
    c(:,2,:) = bias(2) + c(:,2,:);
end
toc

plot(a(1:50), '-x'); hold on; plot(b(1:50),'--o'); plot(c(1:50), ':^')
legend({'you\_raich', 'mine', 'fft'})
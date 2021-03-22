function [res] = wconx(w, opt )
% this function is using pytorch with/out gpu to calculate
% res = conv(x, w), px means pyton x, pker means python kernel
% px has shape of [Batch, F, T], w has shape of [F, T, C]


px = opt.px;
myfunc = opt.myfunc;

if opt.addone
    ww = w(1:end-1, :);
    bias = w(end, :);
else
    ww = w;
end

w = reshape(ww, [opt.F, opt.winsize, opt.C*opt.K]);
ker = single(w);  % for faster computation
pker = py.torch.tensor(py.numpy.array(ker));

r = myfunc.conv(px, pker);  % located in myfunc.py
res = double(r); % back to matlab data, shape of [T, C, n_batch]

if opt.addone
for i= 1:opt.C
    res(:, i, :) = bias(i) + res(:, i, :);
end
end

end
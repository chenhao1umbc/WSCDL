function [res] = wconx(w, opt )
% this function is using pytorch with/out gpu to calculate
% res = conv(x, w), px means pyton x, pker means python kernel
% px has shape of [Batch, F, T], w has shape of [F, T, C]


px = opt.px;
myfunc = opt.myfunc;

w = reshape(w, [opt.F, opt.winsize, opt.C]);
ker = single(w);  % for faster computation
pker = py.torch.tensor(py.numpy.array(ker));

r = myfunc.conv(px, pker);  % located in myfunc.py
res = double(r); % back to matlab data
end
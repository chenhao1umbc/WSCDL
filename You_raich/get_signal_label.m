function [y_hat, P] = get_signal_label(W, X, option)
% this function is written to get the predicted labels in the signal level
% the calculation process is in You's paper
% basically it will get the union of the labels per window

N = size(X, 3);  % number of samples
C = size(W, 2);  % number of classes
y_hat = zeros(N, C);
P = zeros(N, C);
thr = 0.5;
for i = 1:N
    px = py.torch.tensor(py.numpy.array(X(:,:, i)));
    px = px.cuda().float().unsqueeze(py.int(0));
    option.px = px; % added the myfunc to option
    
    [pp,~]=PriorOneBag(W,X(:,:, i),option);  % pp contains null as the last row shape [C+1, n_instance]     [p, ~] = max(pp'); % p is per class probability
    P(i, :) = p(1:10);
    [~, ind] = max(pp);
    pool_label = 1:size(pp, 1)-1;
    label = intersect(pool_label, ind);
    y_hat(i, label) = 1;

end

end  % end of the file



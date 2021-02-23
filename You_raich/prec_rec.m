function [recall, precision] = prec_rec(y, y_hat)
% this function is written to calculate the recall and precision of 
% recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
% precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
% f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));
% meanF1 = @(confusionMat) mean(f1Scores(confusionMat));
stats = confusionmatStats(y(:),y_hat(:));
recall = stats.recall(2);
precision = stats.precision(2);



function [mu trmx] = prepTransform(tvec, comp_count)
% Computes transformation matrix to PCA space
% tvec - training set (one row represents one sample)
% comp_count - count of principal components in the final space
% mu - mean value of the training set
% trmx - transformation matrix to comp_count-dimensional PCA space

mu = mean(tvec);
cmx = cov(tvec); 

%cmx = zeros(size(tvec,2));
%f1 = zeros(size(tvec,1), 1);
%f2 = zeros(size(tvec,1), 1);
%for i=1:size(tvec,2)
%  f1(:,1) = tvec(:,i) - repmat(mu(i), size(tvec,1), 1);
%  cmx(i, i) = f1' * f1;
%  for j=i+1:size(tvec,2)
%    f2(:,1) = tvec(:,j) - repmat(mu(j), size(tvec,1), 1);
%    cmx(i, j) = f1' * f2;
%	cmx(j, i) = cmx(i, j);
%  end
%end
%cmx = cmx / (size(tvec,1)-1);

[evec eval] = eig(cmx);
eval = sum(eval);
[eval evid] = sort(eval);
eval = eval(size(eval,2):-1:1);
evec = evec(:, evid(size(eval,2):-1:1));
% save 'nist_mu.mat' mu
% save 'nist_cov.mat' evec 
trmx = evec(:, 1:comp_count);

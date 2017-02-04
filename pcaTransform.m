function [pcaSet] = pcaTransform(tvec, mu, trmx)
% tvec - matrix containing vectors to be transformed
% mu - mean value of the training set
% trmx - pca transformation matrix
% pcaSet -  outpu set transforrmed to PCA  space

pcaSet = tvec - repmat(mu, size(tvec,1), 1);
%pcaSet = zeros(size(tvec));
%for i=1:size(tvec,1)
%  pcaSet(i,:) = tvec(i,:) - mu;
%end
pcaSet = pcaSet * trmx;

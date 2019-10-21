function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_set = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_set = [0.01 0.03 0.1 0.3 1 3 10 30]';

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

x1 = [1 2 1]; x2 = [0 4 -1]; min = 100;

for i = 1:length(C_set)
  C_i = C_set(i);
  for j = 1:length(sigma_set)
    model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_set(j))); 
    predictions = svmPredict(model, Xval);
    if (mean(double(predictions ~= yval)) < min)
      min = mean(double(predictions ~= yval));
      C = C_i;
      sigma = sigma_set(j);
    endif
  endfor
endfor

% =========================================================================

end

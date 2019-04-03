function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

c_candidates = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_candidates = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% c_candidates = [1; 3];
% sigma_candidates = [1];

trials = [];

for c_i = 1:rows(c_candidates)
    c_current = c_candidates(c_i)
    for s_i = 1:rows(sigma_candidates)
        sigma_current = sigma_candidates(s_i)
        model = svmTrain(X, y, c_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        trials = [trials; [c_current, sigma_current, err]];
    end
end

trials

errors = trials(:, 3)

[min_error, min_index] = min(errors)

C = trials(min_index, 1)
sigma = trials(min_index, 2)

% =========================================================================

end

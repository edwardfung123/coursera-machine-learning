function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% size(y);
for k = 1:num_labels
    y_k = (y == k)';
    predict_k = a3(:,k);
    pos = y_k * log(predict_k);
    neg = (1 - y_k) * log(1 - predict_k);
    J += -pos - neg;
end;
J = J / m + lambda/2/m * (sum(Theta1(:, 2:end)(:).^2) + sum(Theta2(:, 2:end)(:).^2));
% [prob, p] = max(a3, [], 2);

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i = 1:m
    y_i = zeros(num_labels, 1);
    y_i(y(i)) = 1;
    predict_i = a3(i,:)';
    d3 = predict_i - y_i;

    d2 = Theta2' * d3;
    d2 = d2(2:end) .* sigmoidGradient(z2(i, :)');
    Delta2 = Delta2 + d3 * a2(i, :);
    Delta1 = Delta1 + d2 * a1(i, :);
end;

tmp = Theta1;
tmp(:, 1) = 0;
Theta1_grad = Delta1/m + lambda/m * tmp;

tmp = Theta2;
tmp(:, 1) = 0;
Theta2_grad = Delta2/m + lambda/m * tmp;

% backup
% % backpropagate
% d3 = zeros(m, num_labels);
% for k = 1:num_labels
%     y_k = (y == k);
%     predict_k = a3(:, k);
%     d3(:, k) = predict_k - y_k;
% end;
% % size(z2)
% % Theta1
% % d2 = (Theta2' * d3')' % work
% d2 = (d3 * Theta2);
% d2 = d2(:, 2:end); % remove the bias
% d2 = d2 .* sigmoidGradient(z2);

% Delta2 = a2 * d3
% % d2 % correct
% % d3 % correct

% % z2 % correct
% % z3 % assume correct
% % a2 % correct
% % a3 % correct








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

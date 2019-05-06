function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigexp = zeros(size(y)); %exponent for sigmoid
sigexp = X * theta;

% Cost Function
J1 = 0;
J2 = 0; % J = J1 + J2

% Compute J1 first
for i = 1:m
   J1 = J1 + (-y(i) * log(sigmoid(sigexp(i))) - (1 - y(i)) * log(1-sigmoid(sigexp(i))));
endfor
J1 = J1/m;

% Compute J2 next
for j = 1:n
    J2 = J2 + (theta(j)^2);  
endfor
J2 = (J2 * lambda) / (2*m);

% Now the total cost
J = J1 + J2;

% Gradient

% Special for theta(0)
for i = 1:m
    grad(1) = grad(1) + ((sigmoid(sigexp(i)) - y(i)) * X(i,1));
end
grad(1) = grad(1) / m;

% For rest of the thetas, include the lambda term
for j = 2:n
    for i = 1:m
        grad(j) = grad(j) + ((sigmoid(sigexp(i)) - y(i)) * X(i,j));
    end
    grad(j) = (grad(j) / m) + (lambda * theta(j) / m);
end




% =============================================================

end

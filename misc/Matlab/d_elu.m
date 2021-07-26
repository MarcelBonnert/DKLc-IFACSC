function y = d_elu(x, alpha)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    alpha = 1;
end

greater_0_index = x >= 0;
y = zeros(size(x));

y(greater_0_index) = ones(1,sum(greater_0_index));
y(~greater_0_index) = alpha*exp(x(~greater_0_index));

end


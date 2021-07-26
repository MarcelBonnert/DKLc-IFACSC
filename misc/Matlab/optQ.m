function [Q, R] = optQ(Qinit, Rinit, sys, t, A, B, x0,alpha0, alpha1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Q_chol = chol(Qinit)';
R_chol = chol(Rinit)';

Q_chol = Q_chol([1 2 4]);

if nargin < 18
    inputDecoderWeight_cell = 1;
    inputDecoderBias_cell = 1;
end

M0 = [Q_chol(:); R_chol(:)];

optfun = @(x) fun(x, alpha0, alpha1, sys, t, A, B, x0);

opt = optimset('Display', 'iter', 'MaxFunEvals', 10000); % 'Display', 'iter', 
[M, fval] = fminsearch(optfun, M0, opt);

n = size(A, 1);
m = size(B, 2);

M = [M(1:2); 0; M(3:end)];

Q_tri = tril(reshape(M(1:n^2), [n, n]));
R_tri = tril(reshape(M(n^2+1:end), [m, m]));

Q = Q_tri * Q_tri';
R = R_tri * R_tri';

end

function J = fun(x, alpha0, alpha1, sys, t, A, B, x0)

x = [x(1:2); 0; x(3:end)];

n = size(A,1);
m = size(B,2);

Q_tri = tril(reshape(x(1:n^2), [n, n]));
R_tri = tril(reshape(x(n^2+1:end), [m, m]));

Qlqr = Q_tri*Q_tri';
Rlqr = R_tri*R_tri';

C = dlqr(A, B, Qlqr, Rlqr);
    
x = zeros(2, length(t));
x(:,1) = x0;
uvec = zeros(length(t)-1,1);
for k = 2:length(t)
    uc = -C*x(:,k-1);

    uvec(k-1) = uc;

    fun = @(t, x) sys(t, x, uc);

    sol = rkSolver(fun, t(k-1:k), x(:, k-1));
    x(:, k) = sol(:,2);
end

E_u = controlEnergy(uvec, t);

J_x = 0;

for i = 1:n
    J_x = J_x + alpha1(i) * norm(x(i,:), 2)^2/size(x,2);
end

J = alpha0 * E_u(end-1) + J_x;

end

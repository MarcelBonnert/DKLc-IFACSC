function [Q, R] = optKoopmanQ(Qinit, Rinit, sys, t, Kz, Kw, ...
    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, alpha0, ...
    alpha1, selectedStates, ...
    stateEncoderWeight_cell, stateEncoderBias_cell,...
    inputDecoderWeight_cell, inputDecoderBias_cell)
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

optfun = @(x) fun(x, alpha0, alpha1, sys, t, Kz, Kw, ...
    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, selectedStates, ...
    stateEncoderWeight_cell, stateEncoderBias_cell,...
    inputDecoderWeight_cell, inputDecoderBias_cell);

opt = optimset('Display', 'iter', 'MaxFunEvals', 10000); % 'Display', 'iter', 
[M, fval] = fminsearch(optfun, M0, opt);

n = length(selectedStates);
m = size(Kw,2);

M = [M(1:2); 0; M(3:end)];

Q_tri = tril(reshape(M(1:n^2), [n, n]));
R_tri = tril(reshape(M(n^2+1:end), [m, m]));

Qz = Q_tri * Q_tri';
R = R_tri * R_tri';

Q = zeros(size(Kz));

for i = 1:length(selectedStates)
    for j = 1:length(selectedStates)
        Q(selectedStates(i),selectedStates(j)) = Qz(i,j);
    end
end

end

function J = fun(x, alpha0, alpha1, sys, t, Kz, Kw, ...
    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, ...
    selectedStates, stateEncoderWeight_cell, stateEncoderBias_cell,...
    inputDecoderWeight_cell, inputDecoderBias_cell)

x = [x(1:2); 0; x(3:end)];

n = length(selectedStates);
m = size(Kw,2);

Q_tri = tril(reshape(x(1:n^2), [n, n]));
R_tri = tril(reshape(x(n^2+1:end), [m, m]));

Qz = Q_tri*Q_tri';
Rlqr = R_tri*R_tri';

Qlqr = zeros(size(Kz));

for i = 1:length(selectedStates)
    for j = 1:length(selectedStates)
        Qlqr(selectedStates(i),selectedStates(j)) = Qz(i,j);
    end
end

C = dlqr(Kz, Kw, Qlqr, Rlqr);
    
x = zeros(2, length(t));
x(:,1) = x0;
uvec = zeros(length(t)-1,1);
for k = 2:length(t)
    xn = x(:,k-1)./(3*sqrt(normData_x)');
    z = g_z(xn', stateEncoderWeight_cell, stateEncoderBias_cell)';
    
    if length(z) < size(Kz,1)
        z = [xn; z];
    end
    
    uc = -C*(z - z0) + w0;

    if hasNonlinearInput
        un = g_z(uc, inputDecoderWeight_cell, inputDecoderBias_cell);
    else
        un = uc;
    end

    u = un * 3 * sqrt(normData_u);

    uvec(k-1) = u;

    fun = @(t, x) sys(t, x, u);

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

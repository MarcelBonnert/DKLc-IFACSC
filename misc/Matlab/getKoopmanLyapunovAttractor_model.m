function [val, X, val1, val2] = getKoopmanLyapunovAttractor_model(grid, res, Kz, W, b, sys_fun, h, gamma)
%Only for 2D systems

x1 = linspace(grid(1,1), grid(1,2), res(1));
x2 = linspace(grid(2,1), grid(2,2), res(2));

X = [];

for i = 1:res(2)
    X = [X, [x1;x2(i)*ones(1, res(1))]];
end

ne = size(Kz,1);
n_obsv = length(g_z([0 0], W, b));

val = zeros(1, size(X, 2));
val1 = zeros(1, size(X, 2));
val2 = zeros(1, size(X, 2));

if ne > n_obsv
    fun1 = @(x) norm([x' + h * sys_fun(x); g_z(x + h * sys_fun(x)', W, b)'] - (Kz * [x'; g_z(x, W, b)'])', 2);
    fun2 = @(x) gamma * norm([x'; g_z(x, W, b)'],2);
else
    fun1 = @(x) norm(g_z(x + h * sys_fun(x)', W, b) - (Kz * g_z(x, W, b)')', 2);
    fun2 = @(x) gamma * norm(g_z(x, W, b),2);
end
% fun = @(x) norm(g_z(x + h * sys_fun(x)', W, b) - (Kz * g_z(x, W, b)')', 2) < gamma * norm(g_z(x, W, b),2);

for i = 1:size(X, 2)
    val1(i) = fun1(X(:,i)');
    val2(i) = fun2(X(:,i)');
    val(i) = val1(i) < val2(i);
end

end



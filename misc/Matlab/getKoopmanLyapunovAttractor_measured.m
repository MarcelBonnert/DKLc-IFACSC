function [val, val1, val2] = getKoopmanLyapunovAttractor_measured(Kz, W, b, X, gamma)
val = zeros(size(X, 1)-1,1);
val1 = val;
val2 = val;

ne = size(Kz, 1);
nk = length(g_z(X(1,:), W, b));

if nk < ne
    fun1 = @(x1, x2) norm([x2'; g_z(x2, W, b)'] - (Kz * [x1'; g_z(x1, W, b)']), 2);
    fun2 = @(x) gamma * norm(g_z(x, W, b),2);
else
    fun1 = @(x1, x2) norm(g_z(x2, W, b) - (Kz * g_z(x1, W, b)')', 2);
    fun2 = @(x) gamma * norm(g_z(x, W, b),2);
end

for i = 1:size(X, 1)-1
    val1(i) = fun1(X(i,:), X(i+1,:));
    val2(i) = fun2(X(i,:));
    val(i) = val1(i) < val2(i);
end

val = logical(val);

end



function [sub, alpha, gamma] = getStabilitConditions(Kz, P, x, W, b, trajLen)

if nargin < 6
    trajLen = size(x, 2);
end

n = size(Kz, 1);

z = g_z(x, W, b)';

if n > size(z, 1)
    z = [x';z];
end

eps = [];
gk = [];
for i = 1:size(x, 1)/trajLen
    z_loc = z(:, 1 + (i-1) * trajLen:i * trajLen);
    eps = [eps, z_loc(:, 2:end) - Kz * z_loc(:, 1:end - 1)];
    gk = [gk, z_loc(:, 1:end - 1)];
end

fac = sqrt(sum(eps.^2, 1)) ./ sqrt(sum(gk.^2, 1));
sub = max(fac);

Q = dlyap(Kz, P);

alpha = - norm(Kz, 2) + sqrt(norm(Kz, 2)^2 + min(eig(P)) / max(eig(Q)));

if any(abs(eig(Kz)) == 1)
    gamma = -1;
else
    x = -1:0.0001:1;
    y = sqrt(1 - x.^2);

    x = [x, fliplr(x)];
    y = [y, -fliplr(y)];
    
    sigma = -1 * ones(length(x),1);
    for i = 1:length(x)
        sigma(i) = max(svd(inv(complex(x(i), y(i)) * eye(n) - Kz)));
    end

    gamma = 1/max(sigma);
end

end


function [E_u, E_u_s] = controlEnergy(u, t)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

E_u = zeros(size(u));

for i = 2:size(u,1)
    for j = 1:size(u,2)
        E_u(i-1,j) = trapz(t(1:i), (u(1:i,j)).^2);
    end
end

E_u_s = E_u;

E_u = sum(E_u,2);

end


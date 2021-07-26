function tb = get_t_barrier(x, xt, t , tube)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
a = abs(x-xt) < tube;

for i = 1:length(t)
    if all(a(i:end)) == true
        break
    end
end

tb = t(i);

end


function dg_dx_value = dg_dx(x, W, b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~, layer_outputs] = g_z(x, W, b);

dg_dx_value = W{end};

for i = 1:length(W)-1
    dg_dx_value = W{end-i} * diag(d_elu(layer_outputs{end-i})') * dg_dx_value;
end

end


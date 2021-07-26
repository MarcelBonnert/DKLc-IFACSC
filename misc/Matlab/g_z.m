function [z, layers] = g_z(x, W, b)
    
    prevLayer = x;
    layers = cell(length(W),1);
%     layers{1} = prevLayer;
    for i = 1:length(W)-1
        layers{i} = prevLayer*W{i}+b{i};
        prevLayer = elu(layers{i});
%         layers{i+1} = prevLayer;
    end
    
    z = prevLayer*W{end}+b{end};
    layers{end} = z;
end
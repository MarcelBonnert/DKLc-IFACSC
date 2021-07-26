function [mvaf, ms] = det_coef(y,y_hat)
%MVAF(Y,Y_HAT) calculates theMean variance-accounted-for of a data set y 
%and its estimation y_hat

q = min(size(y));

if size(y,1) == q
    y = y';
end

if size(y_hat,1) == q
    y_hat = y_hat';
end

mvaf = 0;

ms = zeros(1, q);

for i = 1:q
    value = (1-sampleVar(y(:,i)-y_hat(:,i))/sampleVar(y(:,i)));
    ms(i) = value;
    mvaf = mvaf + value;
end

mvaf = mvaf/q;
end

function var = sampleVar(x)
x_mean = mean(x);
var = (x-x_mean)'*(x-x_mean)/(length(x)-1);
end
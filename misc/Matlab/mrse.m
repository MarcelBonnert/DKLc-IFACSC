function [mrse, se] = mrse(y,y_hat)
%MRSE(Y,Y_HAT) calculates the mean relative squared error of a data set y 
%and its estimation y_hat

q = min(size(y));

if size(y,1) == q
    y = y';
end

if size(y_hat,1) == q
    y_hat = y_hat';
end

mrse = 0;
for i = 1:q
    sum_diff_sq = sum((y(:,i)-y_hat(:,i)).^2);
    sum_y_sq = sum(y(:,i).^2);
    se(i) = sqrt(sum_diff_sq/sum_y_sq);
    mrse = mrse + se(i);
end

mrse = mrse/q;
end


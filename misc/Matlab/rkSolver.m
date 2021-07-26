function y = rkSolver(y_dot, x, y0)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(y0);
y = zeros(n,length(x));
y(:,1) = y0;
N = length(x)-1;
h = x(2)-x(1);

for i = 1:N
    
    k1 = y_dot(x(i),y(:,i));
    k2 = y_dot(x(i)+.5*h,y(:,i)+.5*k1*h);
    k3 = y_dot(x(i)+.5*h,y(:,i)+.5*k2*h);
    k4 = y_dot(x(i)+h,y(:,i)+k3*h);
    y(:,i+1) = y(:,i)+((k1+2*k2+2*k3+k4)/6)*h;
    
end

end


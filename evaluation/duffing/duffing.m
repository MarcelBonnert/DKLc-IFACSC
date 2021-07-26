function dx = duffing(t, x, u, gamma)
    alpha = 1;
    beta = -1;
%     delta = -0.4;
    delta = 1;
    
    dx = [  x(2);...
            alpha * x(1) + beta * x(1)^3 + gamma * x(2) + delta * u];
end
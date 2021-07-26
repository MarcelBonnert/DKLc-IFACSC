function dx = van_der_pol(t, x, u)
    mu = 1.0;
    beta = 1;
    
    dx = [  x(2);...
            mu * (1 - x(1)^2) * x(2) - x(1) + beta * u];
end
function dx = vanderpol(t,x,u)
    dx =  [ x(2);...
            (1-x(1)^2)*x(2)-x(1)+u];
end
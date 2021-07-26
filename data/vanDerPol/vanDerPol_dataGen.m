%% vanDerPol Autonomous
clear all
len_traj = 300;
Ts = 0.1;
t = 0:Ts:(len_traj-1)*Ts;
fun = @(t,x) van_der_pol(t,x,0);

% percentage traning, validation data
sep = [85, 10];
sep = [sep, 100 - sum(sep)];

N = 2000;
Y = [];
U = [];
T = [];

%% Data generation
for i = 1:N
    
    x0_switch_1 = floor(3*rand(1));

    if any(x0_switch_1 == [1 2])
        x0_switch_2 = round(rand(1));
        
        if x0_switch_2
            x0 = [2.5 * (rand(1)*2 - 1), 6*(round(rand(1))-0.5)]';
        else
            x0 = [2.5 * (2*round(rand(1))-1) , 6 * (rand(1) - 0.5)]';
        end
    else
        angle = rand(1) * 2 * pi;
        x0 = 0.5 * [cos(angle), sin(angle)]';
    end
    
    u = 1*sqrt(1) * randn(length(t),1);
    
    y = zeros(2,length(t));
    xc = x0;
    y(:,1) = xc;
    
    for j = 1:length(t)-1
        fun = @(t,x) van_der_pol(t,x,u(j));
        y_loc = rkSolver(fun, [t(j) t(j + 1)], xc);
        xc = y_loc(:,2);
        y(:,j + 1) = xc;
    end
    
    U = [U; u];
    Y = [Y; y'];
    T = [T; t'];
end

%% Normalization
var_u = var(U);
var_y = var(Y);

min_y = min(Y);
max_y = max(Y);
min_u = min(U);
max_u = max(U);

Un = (U - (max_u + min_u)/2)./(sqrt(var_u)*3);
Yn = (Y - (max_y + min_y)/2)./(sqrt(var_y)*3);

%% Saving
L = size(Y,1);
N_train = L*sep(1)/100 - mod(L*sep(1)/100, len_traj);
N_val = L*sep(2)/100 - mod(L*sep(2)/100, len_traj);
N_test = L*sep(3)/100;

Y_train = Y(1:N_train, :);
Y_val = Y(1 + N_train:N_train + N_val, :);
Y_test = Y(1 + N_train + N_val:end, :);

U_train = U(1:N_train, :);
U_val = U(1 + N_train:N_train + N_val, :);
U_test = U(1 + N_train + N_val:end, :);

T_train = T(1:N_train, :);
T_val = T(1 + N_train:N_train + N_val, :);
T_test = T(1 + N_train + N_val:end, :);

writematrix(Y_train, 'vanDerPol_OszillatorTrainingStates.csv')
writematrix(Y_val, 'vanDerPol_OszillatorValidationStates.csv')
writematrix(Y_test, 'vanDerPol_OszillatorTestStates.csv')

writematrix(U_train, 'vanDerPol_OszillatorTrainingInputs.csv')
writematrix(U_val, 'vanDerPol_OszillatorValidationInputs.csv')
writematrix(U_test, 'vanDerPol_OszillatorTestInputs.csv')

writematrix(T_train, 'vanDerPol_OszillatorTrainingTime.csv')
writematrix(T_val, 'vanDerPol_OszillatorValidationTime.csv')
writematrix(T_test, 'vanDerPol_OszillatorTestTime.csv')












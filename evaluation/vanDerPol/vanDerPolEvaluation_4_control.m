path_full = '../../experiments/vanDerPol/models/';

content = dir(path_full);

runs = {...
        'S1_n_6';... 
        'S2_n_6';... 
        'S3_n_6';... 
        'S4_n_6'}; 
Q = [1 1];
R = .1;

normData_x = [0 0 ;2.1007136568330105, 2.0316690441968492];
normData_u = [0;0.9990524605742528];

t = 0:0.1:20.5;
x0 = [-2 2]';

% Optimization parameters
opt_matrices = false;

alpha0 = [1; 1; 1; 1]*1;
alpha1 = [10 * [1 1];10 * [1 1];10 * [1 1];10 * [1 1]];
    
f1 = figure;
f2 = figure;
f3 = figure;
f4 = figure;
hold on
t5 = zeros(1,2);
for i = 1:length(runs)
    % Read network layers
    Kz = readmatrix([path_full runs{i} ...
            '/K.csv'], 'Delimiter', ' ')';
    Kw = readmatrix([path_full runs{i} ...
            '/L.csv'], 'Delimiter', ' ')';
    
    stateEncoderWeightFiles = dir([path_full runs{i} ...
       '/state_encoderWeights*' ]);
    stateEncoderWeight_cell = {};
    stateEncoderBiasFiles = dir([path_full runs{i} ...
       '/state_encoderBiases*' ]);
    stateEncoderBias_cell = {};
    for j = 1:length(stateEncoderWeightFiles)
       stateEncoderWeight_cell{end+1} = readmatrix([path_full runs{i} ...
           '/' stateEncoderWeightFiles(j).name ], 'Delimiter', ' ');
       stateEncoderBias_cell{end+1} = readmatrix([path_full runs{i} ...
           '/' stateEncoderBiasFiles(j).name ], 'Delimiter', ' ')';
    end
    
    z0 = g_z([0 0], stateEncoderWeight_cell, stateEncoderBias_cell)';
    
    if contains(runs{i}, 'S3') || contains(runs{i}, 'S4')
       hasLinearState = false;
       stateDecoderWeightFiles = dir([path_full runs{i} ...
           '/state_decoderWeights*' ]);
       stateDecoderWeight_cell = {};
       stateDecoderBiasFiles = dir([path_full runs{i} ...
           '/state_decoderBiases*' ]);
       stateDecoderBias_cell = {};
       for j = 1:length(stateDecoderWeightFiles)
           stateDecoderWeight_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               stateDecoderWeightFiles(j).name ], 'Delimiter', ' ');
           stateDecoderBias_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               stateDecoderBiasFiles(j).name ], 'Delimiter', ' ')';
       end
       
       [Qlqr, xi] = getQz([0 0], stateEncoderWeight_cell, ...
           stateEncoderBias_cell, diag(Q), Kz, Kz, t(2));

    else
       hasLinearState = true;
       
       Qlqr = [ diag(Q) zeros(2, size(Kz,2)-2);
                zeros(size(Kz,1)-2, size(Kz,2))];
        z0 = [[0 0]'; z0];
    end

    if contains(runs{i}, 'S2') || contains(runs{i}, 'S4')
        hasNonlinearInput = true;

        inputEncoderWeightFiles = dir([path_full runs{i}, ...
           '/input_encoderWeights*' ]);
        inputEncoderWeight_cell = {};
        inputEncoderBiasFiles = dir([path_full runs{i} ...
           '/input_encoderBiases*' ]);
        inputEncoderBias_cell = {};
        for j = 1:length(inputEncoderWeightFiles)
           inputEncoderWeight_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               inputEncoderWeightFiles(j).name ], 'Delimiter', ' ');
           inputEncoderBias_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               inputEncoderBiasFiles(j).name ], 'Delimiter', ' ')';
        end

        inputDecoderWeightFiles = dir([path_full runs{i} ...
           '/input_decoderWeights*' ]);
        inputDecoderWeight_cell = {};
        inputDecoderBiasFiles = dir([path_full runs{i} ...
           '/input_decoderBiases*' ]);
        inputDecoderBias_cell = {};
        for j = 1:length(inputDecoderWeightFiles)
           inputDecoderWeight_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               inputDecoderWeightFiles(j).name ], 'Delimiter', ' ');
           inputDecoderBias_cell{end+1} = readmatrix(...
               [path_full runs{i} '/' ...
               inputDecoderBiasFiles(j).name ], 'Delimiter', ' ')';
        end
       
        [Rlqr, ~] = getQz(0, inputEncoderWeight_cell, ...
            inputEncoderBias_cell, R, [1]);

        w0 = g_z(0, inputEncoderWeight_cell, inputEncoderBias_cell)';

    else
        hasNonlinearInput = false;
        Rlqr = R;
        w0 = 0;
    end
    
    % Optimize controller if set
    if opt_matrices
        if ~hasLinearState
            if hasNonlinearInput
                [Qlqr, Rlqr] = optKoopmanQ(Qlqr, Rlqr, ...
                    @(t, x, u) vanderpol(t, x, u), t, Kz, Kw, ...
                    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, alpha0(i), ...
                    alpha1(i,:), ...
                    stateEncoderWeight_cell, stateEncoderBias_cell,...
                    inputDecoderWeight_cell, inputDecoderBias_cell);
            else
                [Qlqr, Rlqr] = optKoopmanQ(Qlqr, Rlqr, ...
                    @(t, x, u) vanderpol(t, x, u), t, Kz, Kw, ...
                    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, alpha0(i), ...
                    alpha1(i,:), ...
                    stateEncoderWeight_cell, stateEncoderBias_cell);
            end
            Cs{i,1} = Qlqr;
            Cs{i,2} = Rlqr;
        else
            if hasNonlinearInput
                [Qlqr, Rlqr] = optKoopmanQ(Qlqr, Rlqr, ...
                    @(t, x, u) vanderpol(t, x, u), t, Kz, Kw, ...
                    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, alpha0(i), ...
                    alpha1(i,:), ...
                    stateEncoderWeight_cell, stateEncoderBias_cell,...
                    inputDecoderWeight_cell, inputDecoderBias_cell);
            else
                [Qlqr, Rlqr] = optKoopmanQ(Qlqr, Rlqr, ...
                    @(t, x, u) vanderpol(t, x, u), t, Kz, Kw, ...
                    normData_x, normData_u, hasNonlinearInput, x0, z0, w0, alpha0(i), ...
                    alpha1(i,:), ...
                    stateEncoderWeight_cell, stateEncoderBias_cell);
            end
        end
    end
    
    % Compute controller and simulate
    C = dlqr(Kz, Kw, (Qlqr), (Rlqr));
    
    x = zeros(2, length(t));
    x(:,1) = x0;
    uvec = zeros(length(t)-1,1);
    for k = 2:length(t)
        xn = (x(:,k-1) - normData_x(1,:)')./(3*sqrt(normData_x(2,:))');
        z = g_z(xn', stateEncoderWeight_cell, stateEncoderBias_cell)';
        if hasLinearState
            z = [xn;z];
        end
        
        uc = -C*(z - z0) + w0;
        
        if hasNonlinearInput
            un = g_z(uc, inputDecoderWeight_cell, inputDecoderBias_cell);
        else
            un = uc;
        end
                
        u = un * 3 * sqrt(normData_u(2,:)') + normData_u(1,:)';
        
        uvec(k-1) = u;
        
        fun = @(t, x) vanderpol(t, x, u);
        
        sol = rkSolver(fun, t(k-1:k), x(:, k-1));
        x(:, k) = sol(:,2);
    end

    figure(f1);
    ax(1) = subplot(4,1,1);
    plot(t,x(1,:))
    hold on
    ylabel('x_1')
    ax(2) = subplot(4,1,2);
    plot(t,x(2,:))
    hold on
    ylabel('x_2')
    ax(3) = subplot(4,1,3);
    plot(t(1:end-1),uvec)
    hold on
    ylabel('u')
    ax(4) = subplot(4,1,4);
    Eu = controlEnergy(uvec, t);
    disp(['Eu = ' num2str(Eu(end-1))])
    plot(t(1:end-1),Eu)
    hold on
    ylabel('\int_0^t \Delta \mat{u}\left(\tau\right)^2 d \tau')
    xlabel('Time (s)')
    
    figure(f3);
    plot(t,x(1,:))
    hold on
    ylabel('x_1')
    xlabel('Time (s)')
    
    figure(f4);
    plot(t(1:end-1),Eu)
    hold on
    ylabel('\int_0^t \Delta \mat{u}\left(\tau\right)^2 d \tau')
    xlabel('Time (s)')
  
    figure(f2);
    plot(x(1,:), x(2,:))
    xlabel('x_1')
    ylabel('x_2')
    hold on
    
    t5(i,1) = get_t_barrier(x(1,:), 0, t, 0.1);
    t5(i,2) = get_t_barrier(x(2,:), 0, t, 0.1);
          
    clear stateEncoderWeight_cell stateEncoderBias_cell ...
        stateDecoderWeight_cell stateDecoderBias_cell ...
        inputEncoderWeight_cell inputEncoderBias_cell ...
        inputDecoderWeight_cell inputDecoderBias_cell
     
    
end

figure(f1);
linkaxes(ax, 'x')
xlim([0 10])

%%
alpha0 = 1;
alpha1 = 10 * [1 1];
if opt_matrices
    [Q, R] = optQ(diag(Q), R, @(t, x, u) vanderpol(t, x, u), t, ...
        [0 1;-1 1], [0 1]', x0,alpha0, alpha1, [0 0]', 0, false);
end
Clin = lqr([0 1;-1 1], [0 1]', diag(Q), R);
    
x = zeros(2, length(t));
x(:,1) = x0;
uvec = [0];
for k = 2:length(t)
    u = -Clin*x(:,k-1);
    uvec = [uvec; u];
    fun = @(t, x) vanderpol(t, x, u);

    sol = rkSolver(fun, t(k-1:k), x(:, k-1));
    x(:, k) = sol(:,2);
end

t5(end+1,1) = get_t_barrier(x(1,:), 0, t, 0.1);
t5(end,2) = get_t_barrier(x(2,:), 0, t, 0.1)

figure(f1);
% figure
subplot(4,1,1)
% hold on
plot(t,x(1,:)')
subplot(4,1,2)
% hold on
plot(t,x(2,:)')
subplot(4,1,3)
% hold on
plot(t,uvec)
subplot(4,1,4)
% hold on
Eu = controlEnergy(uvec, t);
plot(t(1:end-1),Eu(1:end-1))

figure(f3);
plot(t,x(1,:))
hold on
ylabel('x_1')
xlabel('Time (s)')

figure(f4);
plot(t(1:end-1),Eu(1:end-1))
hold on
ylabel('\int_0^t \Delta \mat{u}\left(\tau\right)^2 d \tau')
xlabel('Time (s)')
xlim([0 10])

figure(f2)
plot(x(1,:)',x(2,:)')














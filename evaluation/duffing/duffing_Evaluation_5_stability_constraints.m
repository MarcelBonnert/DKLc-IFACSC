load('normalizationData.mat')
normData_x_a3 = [[0 0]; normData_x_a1(2,:)];

runs = {
        'S3_n_7';...
        };     
    
data_path = '../../data/duffing/duffingOscillator_uncorrectedTestStates.csv';
path_full = '../../experiments/duffing/model/';

AP = 0;

if AP == -1    
    Q = 1*[11 0 0; 15 0 0; 14 0 0; 600 0 0];% [-1 -1] -> [-1 0]
    R = [1 1 2 1];
    normData_x = normData_x_a1;
    ap_x = 1*[-1 0 -1];
elseif AP == 1
    Q = 1*[20 0 0; 20 0 0; 50 0 0; 15 0 0]; % [-1 -1] -> [1 0]
    R = [1 1 1 1];
    normData_x = normData_x_a2;
    ap_x = -1*[-1 0 -1];
else
    Q = 1*[10 0 0; 10 0 0; 100 0 0; 10 0 0];% [-1 -1] -> [0 0]
    R = [1 1 1 1];
    normData_x = normData_x_a3;
    ap_x = 0*[-1 0 -1];
end

ap_u = [0];

t = 0:0.2:40.4;
x0 = [-2 -2]';
vals = [];

y = rkSolver(@(t,x) duffing(t,x,0, -0.4), 0:0.2:20, x0);

xn_1 = (y - normData_x(1,:)')./(3*sqrt(normData_x(2,:))') + normData_x(1,:)';

y = rkSolver(@(t,x) duffing(t,x,0, -0.4), 0:0.2:20, -x0);

xn_2 = (y - normData_x(1,:)')./(3*sqrt(normData_x(2,:))') + normData_x(1,:)';

for i = 1:length(runs)
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
    
    z0 = g_z(ap_x, stateEncoderWeight_cell, stateEncoderBias_cell)';
    
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
        
       [~, selectedStates, identityError, identityErrorMatrix, ...
            Qlqr, xi, selected_eigenfunc, Qzg3] = ...
            getQz(ap_x, ...
                stateEncoderWeight_cell, stateEncoderBias_cell, ...
                stateDecoderWeight_cell, stateDecoderBias_cell, ...
                diag(Q(i,:)), 2, Kz, Kw, 0.2);
        
    else
       hasLinearState = true;
       
       Qlqr = [ diag(Q(i,:)) zeros(3, size(Kz,2)-3);
                zeros(size(Kz,1)-3, size(Kz,2))];
        z0 = [ap_x'; z0];
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
       
         [Rlqr, selectedInputs, identityError_u, identityErrorMatrix_u] = ...
            getQz(ap_u, ...
            inputEncoderWeight_cell, inputEncoderBias_cell, ...
            inputDecoderWeight_cell, inputDecoderBias_cell, ...
            R(i), 1);
        
        w0 = g_z(ap_u, inputEncoderWeight_cell, inputEncoderBias_cell)';
          
    else
        hasNonlinearInput = false;
        Rlqr = R(i);
        w0 = 0;
    end
    
    C = dlqr(Kz, Kw, Qlqr, Rlqr);
    
    res = 500;
    range = 1.5;
    
    if contains(runs{i}, 'S3') || contains(runs{i}, 'S4')
        if contains(runs{i}, 'S2') || contains(runs{i}, 'S4')
            controller_equation = @(x) g_z(-C * g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)', ...
                inputDecoderWeight_cell, inputDecoderBias_cell);
        else
            controller_equation = @(x) -C * g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)';
        end
        
        mat = readmatrix(data_path);

        % without controller
        [sub, alpha, gamma] = getStabilitConditions(Kz, eye(size(Kz,1)), ...
            mat, stateEncoderWeight_cell, stateEncoderBias_cell, 300);
        vals = [vals, [sub; alpha; gamma]];
        
        [val, X, val1, val2] = ...
            getKoopmanLyapunovAttractor_duffing([-range, range; -range range], ...
            [res res], Kz, stateEncoderWeight_cell, stateEncoderBias_cell, ...
            @(x) duffing(0, x, 0, -0.4), 0.2, alpha,AP);

        [X_mesh, Y_mesh] = meshgrid(linspace(-range, range, res), linspace(-range, range, res));

        Z = zeros(res, res);

        for j = 1:size(X,2)
            if val(j)
                index = (X_mesh == X(1,j)) & (Y_mesh == X(2,j));
                Z(index) = 1;
            end
        end

        index_uncontrolled = Z == 0;

        % with controller
        [sub, alpha, gamma] = getStabilitConditions(Kz-Kw * C, eye(size(Kz,1)), ...
            mat, stateEncoderWeight_cell, stateEncoderBias_cell, 300);
        disp(alpha)
        disp(gamma)
        
        [val, X, val1, val2] = ...
            getKoopmanLyapunovAttractor_duffing([-range, range; -range range], ...
            [res res], Kz - Kw * C, stateEncoderWeight_cell, stateEncoderBias_cell, ...
            @(x) duffing(0, x, controller_equation(x),-0.4), ...
            0.2, max([alpha, gamma]), AP);

        [X_mesh, Y_mesh] = meshgrid(linspace(-range, range, res), linspace(-range, range, res));

        Z = zeros(res, res);

        for j = 1:size(X,2)
            if val(j)
                index = (X_mesh == X(1,j)) & (Y_mesh == X(2,j));
                Z(index) = 1;
            end
        end

        for j = 1:size(xn,2)
            index = (min(min(abs(X_mesh - xn(1,j)))) == abs(X_mesh - xn(1,j))) & ...
                (min(min(abs(Y_mesh - xn(2,j)))) == abs(Y_mesh - xn(2,j))); %(Y_mesh == xn(2,j));
            Z(index) = 2;
        end

        Z(index_uncontrolled) = 0.5;
%%
        figure
        imagesc(linspace(-range, range, res), linspace(-range, range, res), Z);

        Col = [90 90 90;0 143 196]/255;
        colormap(Col);

        set(gca,'ColorScale','linear')
        xlabel('x_1 (-)')
        ylabel('x_2 (-)')
        ax = gca;
        ax.FontName = 'Times';
        ax.FontSize = 14;
        hold on
        plot(xn_1(1,:), xn_1(2,:), 'w', 'linewidth', 2)
        plot(xn_2(1,:), xn_2(2,:), 'w', 'linewidth', 2)

        hold on
        plot(xn(1,:), xn(2,:), 'm', 'linewidth', 2)

        a = 1;
    else
        mat = readmatrix(data_path);
        [sub, alpha, gamma] = getStabilitConditions(Kz-Kw * C, eye(size(Kz,1)), ...
            mat, stateEncoderWeight_cell, stateEncoderBias_cell, 300);
        disp(alpha)
        disp(gamma)
    end
end













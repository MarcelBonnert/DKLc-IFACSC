runs = {'S3_n_8'}; 
    
data_path = '../../data/vanDerPol/vanDerPol_OszillatorTestStates.csv';
path_full = '../../experiments/vanDerPol/models/';

Q = [1 1; 1 1; 1 1; 1 1];
R = 1;

normData_x = [0 0 ;2.1007136568330105, 2.0316690441968492];
normData_u = [0;0.9990524605742528];

t = 0:0.1:20.5;
x0 = [-2 2]';
vals = [];

resolution = 500;
range = 1;

y = rkSolver(@(t,x) vanderpol(t,x,0), 0:0.1:10, [-1.155 1.033]);

xn = (y - normData_x(1,:)')./(3*sqrt(normData_x(2,:))');

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
       [Qlqr, xi] = getQz([0 0], stateEncoderWeight_cell, stateEncoderBias_cell,...
           diag(Q(i,:)), Kz, Kw, 0.1);
        
    else
       hasLinearState = true;
       
       Qlqr = [ diag(Q(i,:)) zeros(2, size(Kz,2)-2);
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
       [Rlqr, xi] = getQz(0, inputEncoderWeight_cell, inputEncoderBias_cell,...
           R, Kz, Kw, 0.1);
        
        w0 = g_z(0, inputEncoderWeight_cell, inputEncoderBias_cell)';
          
    else
        hasNonlinearInput = false;
        Rlqr = R;
        w0 = 0;
    end
    
    C = dlqr(Kz, Kw, Qlqr, Rlqr);
    
    if contains(runs{i}, 'S3')
        cont = @(x) -C * g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)';
    elseif contains(runs{i}, 'S4')
        cont = @(x) g_z(-C * g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)', ...
            inputDecoderWeight_cell, inputDecoderBias_cell);
    elseif contains(runs{i}, 'S2')
        cont = @(x) g_z(-C * [x'; g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)'], ...
            inputDecoderWeight_cell, inputDecoderBias_cell);
    else
        cont = @(x) -C * [x'; g_z(x, stateEncoderWeight_cell, stateEncoderBias_cell)'];
    end

    mat = readmatrix(data_path);

    % without controller
    [sub, alpha, gamma] = getStabilitConditions(Kz, eye(size(Kz,1)), ...
        mat, stateEncoderWeight_cell, stateEncoderBias_cell, 300);
    vals = [vals, [sub; alpha; gamma]];

    [val, X, val1, val2] = ...
        getKoopmanLyapunovAttractor([-range, range; -range range], ...
        [resolution resolution], Kz, stateEncoderWeight_cell, stateEncoderBias_cell, ...
        @(x) vanderpol(0, x, 0), 0.1, alpha);

    [X_mesh, Y_mesh] = meshgrid(linspace(-range, range, resolution), linspace(-range, range, resolution));

    Z = zeros(resolution, resolution);

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

    [val, X, val1, val2] = ...
        getKoopmanLyapunovAttractor([-range, range; -range range], ...
        [resolution resolution], Kz - Kw * C, stateEncoderWeight_cell, stateEncoderBias_cell, ...
        @(x) vanderpol(0, x, cont(x)), ...
        0.1, alpha);

    [X_mesh, Y_mesh] = meshgrid(linspace(-range, range, resolution), linspace(-range, range, resolution));

    Z = zeros(resolution, resolution);

    for j = 1:size(X,2)
        if val(j)
            index = (X_mesh == X(1,j)) & (Y_mesh == X(2,j));
            Z(index) = 1;
        end
    end

    Z(index_uncontrolled) = 0.5;

    figure
    imagesc(linspace(-range, range, resolution), linspace(-range, range, resolution), Z);
    Col = copper();
    colormap(Col);
    colorbar();
    set(gca,'ColorScale','log')
    xlabel('x_1')
    ylabel('x_2')
    hold on
    plot(xn(1,:), xn(2,:), 'w', 'linewidth', 2)

end







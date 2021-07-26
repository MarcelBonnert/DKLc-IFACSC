clear all
% TODO insert path with model file (same as inserted in params['model_path'])
path_full = '../../experiments/vanDerPol/models/'

content = dir(path_full);

% TODO insert length of trajectories
length_traj = 100

pred_h = 50;

% TODO insert path to the evaluation data
saving_path = ['predicted_data/p' num2str(length_traj) '/']

% TODO insert number of nonlinear states
num_states = 2

% TODO insert number of inputs
num_inputs = 1

% TODO set certain indicies to true to plot certain runs
plot_results = false(length(content)-2,1);

mrse_runs_pred_horizon = [];
mrse_runs_ext_pred_horizon = [];

for j = 1:length(content)-2
    disp('# # # # # # # # # # # # # # # # # # # #')
    disp(content(j + 2).name)
    disp('# # # # # # # # # # # # # # # # # # # #')
    
    path = [path_full, content(j + 2).name, '/'];
    K = load([path, 'K.csv'])' ;
    L = load([path, 'L.csv'])';
    eigenvalues_pred = complex(eig(K));
    G_pred = zpk([], eigenvalues_pred, 1);

    % Plot Eigenvalues
    phi = (0:0.01:2.4 * pi);
    circle = [cos(phi); sin(phi)];
    if plot_results(j)
        figure;
        plot(eigenvalues_pred, 'rx')
        title(content(j + 2).name)
        hold on;
        grid on;
        plot(circle(1, :),circle(2, :), 'k-')
        xlabel('Realteil')
        ylabel('Imaginärteil')
        set(gca, 'xLim', [-0.5; 1.5])
        set(gca, 'yLim', [-1; 1])
        set(gca,'DataAspectRatio',[1 1 1])
    end
    
    scale_state = ones(1, num_states);

%% Plot input encoder
    inputs = load([saving_path, content(j + 2).name, '/', 'TestTraj_input.csv']);
    inputs_pred = load([saving_path, content(j + 2).name, '/', 'TestTraj_input_pred.csv']);
    
    if plot_results(j)
        for ind = (1: num_inputs)
            figure;
            subplot(2, 1, 1)
            plot((0:length(inputs(:, ind))-1), inputs(:, ind))
            hold on
            plot((0:length(inputs_pred(:, ind))-1), inputs_pred(:, ind))
            grid on;
            legend('true input', 'estimated input')
            ylabel(['Input ', string(ind)])
            title(content(j + 2).name)
            subplot(2, 1, 2)
            plot((0:length(inputs(:, ind))-1), inputs_pred(:, ind) - inputs(:, ind))
            grid on;
            ylabel('absolute error')
            xlabel('{\it k}')
        end
    end

    %% plot histogram
    states = load([saving_path, content(j + 2).name, '/', 'TestTraj_refTotal.csv']);
    if contains(content(j + 2).name, 'S3') || contains(content(j + 2).name, 'S4')
        states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajTotal.csv']);
    else
        states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
    end
    w = size(states);
    scale = 1;

    for ind1 = (1:num_states)
        histo1 = HistogramTimeline((-1.0:0.01:1.0), length_traj, 1);
        for ind = (0:w(1)-1)
            histo1.addData((states(ind+1, ind1) - states_pred(ind+1, ind1))/scale * scale_state(ind1), mod(ind, length_traj)+1);
        end
        
        if plot_results(j)
            figure
            histo1.plot(pred_h)
            xlabel('{\it k}')
            ylabel(['Zustand ', string(ind1)])
            title(content(j + 2).name)
        end

        [mu_prog, var_prog] = histo1.getMeanVar();

    end
   
    % Evaluate error measures
    if j == 1
        measures_full = information_criteria(states(:,1:num_states), states_pred(:,1:num_states), length(eigenvalues_pred));

        measures_full(1, end + 1) = {'n'};
        measures_full{2, end} = length(eigenvalues_pred);
        
        measures_full = [{'';0} measures_full];
        measures_full{1,1} = 'Run';
        
        measures_full{2,1} = content(j + 2).name;
        
        measures_mean = measures_full(1,[1 2 3 4 5 7]);
        measures_pred_mean = measures_full(1,[1 2 3 4 5 7]);
        
        mrse_runs_pred_horizon = [];
        mrse_runs_ext_pred_horizon = [];
        
    else
        measures_full_run = information_criteria(states(:,1:num_states), states_pred(:,1:num_states), length(eigenvalues_pred));
        
        measures_full(end+1, :) = [{content(j + 2).name} measures_full_run(2,:) {length(eigenvalues_pred)}];
    end
    
    mrse_run_pred_horizon = [];
    mrse_run_len_traj_horizon = [];
    
    measures_pred_horizon = zeros(size(states,1)/pred_h, 4);
    measures_len_traj_horizon = zeros(size(states,1)/length_traj, 4);

    mse_mat = zeros(size(states,1)/length_traj, num_states);
    det_coef_mat = zeros(size(states,1)/length_traj, num_states);

    for k = 1:size(states,1)/length_traj-1
        IC_info_pred = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj+pred_h, 1:num_states), ...
            states_pred(1+(k-1)*length_traj:(k-1)*length_traj+pred_h,1:num_states), length(eigenvalues_pred));
        measures_pred_horizon(k,:) = cell2mat(IC_info_pred(2,1:4));
        
        mrse_run_pred_horizon = [mrse_run_pred_horizon; IC_info_pred{end,end}];
        
        IC_info_len_traj = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj+length_traj, 1:num_states), ...
            states_pred(1+(k-1)*length_traj:(k-1)*length_traj+length_traj,1:num_states), length(eigenvalues_pred));
        measures_len_traj_horizon(k,:) = cell2mat(IC_info_len_traj(2,1:4));
        
        mrse_run_len_traj_horizon = [mrse_run_len_traj_horizon; IC_info_len_traj{end,end}];
        
        [~,mse_single] = mrse(states(1+(k-1)*length_traj:(k-1)*length_traj+pred_h, 1:num_states), ...
            states_pred(1+(k-1)*length_traj:(k-1)*length_traj+pred_h,1:num_states));

        [~,det_coef_single] = det_coef(states(1+(k-1)*length_traj:(k-1)*length_traj+pred_h, 1:num_states), ...
            states_pred(1+(k-1)*length_traj:(k-1)*length_traj+pred_h,1:num_states));

        mse_mat(k, :) = mse_single.^2;
        det_coef_mat(k, :) = det_coef_single;

    end


    mrse_runs_pred_horizon = [mrse_runs_pred_horizon; mean(mrse_run_pred_horizon)*100];
    mrse_runs_ext_pred_horizon = [mrse_runs_ext_pred_horizon; mean(mrse_run_len_traj_horizon)*100];

    measures_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(measures_pred_horizon, 1)), {length(eigenvalues_pred)}];
    measures_pred_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(measures_len_traj_horizon, 1)), {length(eigenvalues_pred)}];

    disp('----------------------------------------------------')
    disp('---------------- Nonlinear States ------------------')
    disp(['--------------- Order: ' num2str(size(K,1)) ' -------------------'])
    disp('----------------------------------------------------')
    disp('Mean mse')
    disp(mean(sqrt(mse_mat))*100)
    disp(mean(mean(sqrt(mse_mat)))*100)
    disp('Mean R^2')
    disp(mean(det_coef_mat)*100)
    disp(mean(mean(det_coef_mat))*100)
    
    %% Plot linear error historgram
    states = load([saving_path, content(j + 2).name, '/', 'TestTrajLin_refTotal.csv']);
    states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
    w = size(states);
    scale = 1;
    if plot_results(j)
        figure;
    end
    
    for state_nmbr = (1:2)
        histo1_lin = HistogramTimeline((-0.2:0.01:0.2), length_traj, 1);
        scale = std(states(:, state_nmbr)) * 3;
        for ind = (0:w(1)-1)
            %scale = states(ind+1, state_nmbr) + 1e-4;
            histo1_lin.addData((states(ind+1, state_nmbr) - states_pred(ind+1, state_nmbr))/scale, mod(ind, length_traj)+1);
        end
        
        if plot_results(j)
            subplot(1, 2, state_nmbr)
            histo1_lin.plot(50)
            xlabel('{\it k}')
            ylabel('Δ{\it y}_{R, ' + string(state_nmbr)+'}')
            title(content(j + 2).name)
        end
    end
    %% Plot observables
    states = load([saving_path, content(j + 2).name, '/', 'TestTrajLin_refTotal.csv']);
    states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
    w = size(states);
    scale = 1;
    if plot_results(j)
        figure;
    end

    obsv = size(states,2);

    mse_lin_mat = zeros(size(states,1)/length_traj, num_states);
    det_coef_lin_mat = zeros(size(states,1)/length_traj, num_states);

    for state_nmbr = (1:obsv)
        histo1_lin = HistogramTimeline((-1.0:0.01:1.0), length_traj, 1);
        scale = std(states(:, state_nmbr)) * 3;
        for ind = (0:w(1)-1)
            histo1_lin.addData((states(ind+1, state_nmbr) - states_pred(ind+1, state_nmbr))/scale, mod(ind, length_traj)+1);
        end
        for k = 1:size(states,1)/length_traj-1
            [~,mse_single] = mrse(states(1+(k-1)*length_traj:(k-1)*length_traj+length_traj, state_nmbr), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+length_traj,state_nmbr));

            [~,det_coef_single] = det_coef(states(1+(k-1)*length_traj:(k-1)*length_traj+length_traj, state_nmbr), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+length_traj,state_nmbr));

            mse_lin_mat(k, state_nmbr) = mse_single.^2;
            det_coef_lin_mat(k, state_nmbr) = det_coef_single;
        end
%             
        if plot_results(j)
            subplot(1, obsv, state_nmbr)
            histo1_lin.plot(pred_h)
            xlabel('{\it k}')
            ylabel('Δ{\it y}_{R, ' + string(state_nmbr)+'}')
            title(content(j + 2).name)
        end
    end
    
    disp('----------------------------------------------------')
    disp('------------------ Linear States -------------------')
    disp('----------------------------------------------------')
    disp('Mean mse lin')
    disp(mean(sqrt(mse_lin_mat))*100)
    disp(mean(mean(sqrt(mse_lin_mat)))*100)
    disp('Mean R^2 lin')
    disp(mean(det_coef_lin_mat)*100)
    disp(mean(mean(det_coef_lin_mat))*100)

end
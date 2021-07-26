% close all
clear all

% TODO insert path with model file (same as inserted in params['model_path'])
path_full = '../../experiments/duffing/model/'

content = dir(path_full);

% TODO insert length of trajectories
length_traj = 100
 
% TODO insert path where to put the evaluation data
saving_path = ['duffing/duffing/p' num2str(length_traj) '_uncorrected_added_states_full/']

% TODO insert number of nonlinear states
num_states = 3

% TODO insert number of inputs
num_inputs = 1

% TODO set certain indicies to true to plot certain runs
plot_results = true(length(content)-2,1);

mrse_runs_50 = [];
mrse_runs_100 = [];
mrse_lin_runs_50 = [];
mrse_lin_runs_100 = [];

syms x1 x2 u1 real
x = [x1 x2]';
dx = duffing(0, x, u1, -0.4);

A = expm(double(subs(jacobian(dx,x),[x;u1],[-1 0 0]'))*0.2);

eig_ref = eig(A);
fig_eigs = figure;

% Sort values
cuttingIndicies = 1:100;

for i = 1:99
    cuttingIndicies = [cuttingIndicies, (-99:100)+i*3*100];
end

cuttingIndicies = cuttingIndicies';

mrse_oa = [];

leg = cell(4,1);
kk = 1;
for j = 1:length(content)-2
     if plot_results(j)
        disp('# # # # # # # # # # # # # # # # # # # #')
        disp(content(j + 2).name)
        disp('# # # # # # # # # # # # # # # # # # # #')
    
        path = [path_full, content(j + 2).name, '/'];
        K = load([path, 'K.csv'])' ;
        L = load([path, 'L.csv'])';
        eigenvalues_pred = complex(eig(K));
        G_pred = zpk([], eigenvalues_pred, 1);

        phi = (0:0.01:2.4 * pi);
        circle = [cos(phi); sin(phi)];
        if plot_results(j)
            figure;
            plot(eigenvalues_pred, 'rx')
            title(content(j + 2).name)
            hold on;
            grid on;
            plot(real(eig_ref), imag(eig_ref), 'bo')
            plot(circle(1, :),circle(2, :), 'k-')
            xlabel('Realteil')
            ylabel('Imaginärteil')
            set(gca, 'xLim', [-0.5; 1.5])
            set(gca, 'yLim', [-1; 1])
            set(gca,'DataAspectRatio',[1 1 1])
            
            figure(fig_eigs)
            plot(eigenvalues_pred, 'rx')
            title(content(j + 2).name)
            hold on;
            leg{kk} = content(j + 2).name;
            kk = kk + 1;
        end
        [eigenv, D] = eig(K);
        diag(D);
        
        scale_state = ones(1, num_states);

        %% plot histogram
        states = load([saving_path, content(j + 2).name, '/', 'TestTraj_refTotal.csv']);
        states_lin = load([saving_path, content(j + 2).name, '/', 'TestTraj_refTotal.csv']);
        if contains(content(j + 2).name, 'S3') ||contains(content(j + 2).name, 'S4')
            states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajTotal.csv']);
        else
            states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
        end
        states_lin_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
        
        states = states(cuttingIndicies,:);
        states_lin = states_lin(cuttingIndicies,:);
        states_pred = states_pred(cuttingIndicies,:);
        states_lin_pred = states_lin_pred(cuttingIndicies,:);
        
        w = size(states);
        scale = 1;
        
%         mse = zeros(num_states,100);
        
        for ind1 = (1:num_states - 1)
            histo1 = HistogramTimeline((-1.0:0.01:1.0), length_traj, 1);
            for ind = (0:w(1)-1)
                histo1.addData((states(ind+1, ind1) - states_pred(ind+1, ind1))/scale * scale_state(ind1), mod(ind, length_traj)+1);
            end

            if plot_results(j)
                figure
                histo1.plot(50)
                xlabel('{\it k}')
                ylabel(['Zustand ', string(ind1)])
                title(content(j + 2).name)
            %     xlim([0 length_traj-10])
            %     ylim([-1 1])
            end

            [mu_prog, var_prog] = histo1.getMeanVar();
%             mse(ind1,:) = var_prog + mu_prog.^2;

        end

        % todo. mittel über alle bei single und prediction --> neue mesaures

        if j == 1
            measures_single = information_criteria(states(1:50,1:num_states - 1), states_pred(1:50,1:num_states - 1), length(eigenvalues_pred));
            measures_single_pred = information_criteria(states(1:length_traj,1:num_states - 1), states_pred(1:length_traj,1:num_states - 1), length(eigenvalues_pred));
            measures_full = information_criteria(states(:,1:num_states - 1), states_pred(:,1:num_states - 1), length(eigenvalues_pred));

            measures_single(1, end + 1) = {'n'};
            measures_single_pred(1, end + 1) = {'n'};
            measures_full(1, end + 1) = {'n'};
            measures_single{2, end} = length(eigenvalues_pred);
            measures_single_pred{2, end} = length(eigenvalues_pred);
            measures_full{2, end} = length(eigenvalues_pred);

            measures_single = [{'';0} measures_single];
            measures_single{1,1} = 'Run';
            measures_single_pred = [{'';0} measures_single_pred];
            measures_single_pred{1,1} = 'Run';
            measures_full = [{'';0} measures_full];
            measures_full{1,1} = 'Run';

            measures_single{2,1} = content(j + 2).name;
            measures_single_pred{2,1} = content(j + 2).name;
            measures_full{2,1} = content(j + 2).name;

            measures_single_mean = measures_single(1,[1 2 3 4 5 7]);
            measures_single_pred_mean = measures_single_pred(1,[1 2 3 4 5 7]);
            
            measures_lin_single_mean = measures_single(1,[1 2 3 4 5 7]);
            measures_lin_single_pred_mean = measures_single_pred(1,[1 2 3 4 5 7]);

        else
            measures_single_run = information_criteria(states(1:50,1:num_states - 1), states_pred(1:50,1:num_states - 1), length(eigenvalues_pred));
            measures_single_pred_run = information_criteria(states(1:length_traj,1:num_states - 1), states_pred(1:length_traj,1:num_states - 1), length(eigenvalues_pred));
            measures_full_run = information_criteria(states(:,1:num_states - 1), states_pred(:,1:num_states - 1), length(eigenvalues_pred));

            measures_single(end+1, :) = [{content(j + 2).name} measures_single_run(2,:) {length(eigenvalues_pred)}];
            measures_single_pred(end+1, :) = [{content(j + 2).name} measures_single_pred_run(2,:) {length(eigenvalues_pred)}];
            measures_full(end+1, :) = [{content(j + 2).name} measures_full_run(2,:) {length(eigenvalues_pred)}];
        end

        mrse_run_50 = [];
        mes_single_run = zeros(size(states,1)/length_traj, 4);
        
        mse_mat = zeros(size(states,1)/length_traj, num_states - 1);
        mvaf_mat = zeros(size(states,1)/length_traj, num_states - 1);
        
        mrse_oa_runs = [];
        
        for k = 1:size(states,1)/length_traj-1
            a = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:num_states - 1), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:num_states - 1), length(eigenvalues_pred));
            mes_single_run(k,:) = cell2mat(a(2,1:4));
            
            [~,mse_single] = mrse(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:num_states - 1), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:num_states - 1));
        
            [~,mvaf_single] = mvaf(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:num_states - 1), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:num_states - 1));
            
            mvaf_single(isinf(mvaf_single)) = 1;
            
            mrse_oa_runs = [mrse_oa_runs;mse_single];
            
            mse_mat(k, :) = mse_single.^2;
            mvaf_mat(k, :) = mvaf_single;
        
            mrse_run_50 = [mrse_run_50; a{end,end}];
        end
        
        mrse_oa = [mrse_oa;mean(mrse_oa_runs)];
        
        disp('Mean mse')
        disp(mean(mse_mat)*100)
        disp(mean(mean(mse_mat))*100)
        disp('Mean R^2')
        disp(mean(mvaf_mat)*100)
        disp(mean(mean(mvaf_mat))*100)
        
        mrse_run_100 = [];
        mes_single_pred_run = zeros(size(states,1)/length_traj, 4);
        for k = 1:size(states,1)/length_traj-1
            a = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj + length_traj, 1:num_states - 1), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj + length_traj,1:num_states - 1), length(eigenvalues_pred));
            mes_single_pred_run(k,:) = cell2mat(a(2,1:4));
            mrse_run_100 = [mrse_run_100; a{end,end}];
        end

        mrse_runs_50 = [mrse_runs_50; mean(mrse_run_50)*100];
        mrse_runs_100 = [mrse_runs_100; mean(mrse_run_100)*100];

        measures_single_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(mes_single_run, 1)), {length(eigenvalues_pred)}];
        measures_single_pred_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(mes_single_pred_run, 1)), {length(eigenvalues_pred)}];

        %%
        states = load([saving_path, content(j + 2).name, '/', 'TestTrajLin_refTotal.csv']);
        states_pred = load([saving_path, content(j + 2).name, '/', 'TestTrajLinTotal.csv']);
        
        states = states(cuttingIndicies,:);
        states_pred = states_pred(cuttingIndicies,:);
        
        w = size(states);
        scale = 1;
        if plot_results(j)
            figure;
        end

        obsv = size(states,2);
        
        mse_lin_mat = zeros(size(states,1)/length_traj, num_states - 1);
        mvaf_lin_mat = zeros(size(states,1)/length_traj, num_states - 1);
        
        for state_nmbr = (1:obsv)
            histo1_lin = HistogramTimeline((-1.0:0.01:1.0), length_traj, 1);
            scale = std(states(:, state_nmbr)) * 3;
            for ind = (0:w(1)-1)
                %scale = states(ind+1, state_nmbr) + 1e-4;
                histo1_lin.addData((states(ind+1, state_nmbr) - states_pred(ind+1, state_nmbr))/scale, mod(ind, length_traj)+1);
            end
            for k = 1:size(states,1)/length_traj-1
                [~,mse_single] = mrse(states(1+(k-1)*length_traj:(k-1)*length_traj+length_traj, state_nmbr), ...
                    states_pred(1+(k-1)*length_traj:(k-1)*length_traj+length_traj,state_nmbr));

                [~,mvaf_single] = mvaf(states(1+(k-1)*length_traj:(k-1)*length_traj+length_traj, state_nmbr), ...
                    states_pred(1+(k-1)*length_traj:(k-1)*length_traj+length_traj,state_nmbr));
            
                mvaf_single(isinf(mvaf_single)) = 1;
                
                mse_lin_mat(k, state_nmbr) = mse_single.^2;
                mvaf_lin_mat(k, state_nmbr) = mvaf_single;
            end
%             
            if plot_results(j)
                subplot(1, obsv, state_nmbr)
                histo1_lin.plot(50)
                xlabel('{\it k}')
                ylabel('Δ{\it y}_{R, ' + string(state_nmbr)+'}')
                title(content(j + 2).name)
            end
        end
        
        disp('Mean mse lin')
        disp(mean(mse_lin_mat)*100)
        disp(mean(mean(mse_lin_mat))*100)
        disp('Mean R^2 lin')
        disp(mean(mvaf_lin_mat)*100)
        disp(mean(mean(mvaf_lin_mat))*100)
        
        % % % % % % % % % %
        % % Analyse aicc % %
        % % % % % % % % % %
        mrse_lin_run_50 = [];
        mes_lin_single_run = zeros(size(states,1)/length_traj, 4);
        
        mse_lin_mat = zeros(size(states,1)/length_traj, length(eigenvalues_pred));
        mvaf_lin_mat = zeros(size(states,1)/length_traj, length(eigenvalues_pred));
        
        for k = 1:size(states,1)/length_traj-1
            a = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:length(eigenvalues_pred)), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:length(eigenvalues_pred)), length(eigenvalues_pred));
            mes_lin_single_run(k,:) = cell2mat(a(2,1:4));
            
            [~,mse_single] = mrse(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:length(eigenvalues_pred)), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:length(eigenvalues_pred)));
        
            [~,mvaf_single] = mvaf(states(1+(k-1)*length_traj:(k-1)*length_traj+50, 1:length(eigenvalues_pred)), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj+50,1:length(eigenvalues_pred)));
            
            mvaf_single(isinf(mvaf_single)) = 1;
            
            mse_lin_mat(k, :) = mse_single.^2;
            mvaf_lin_mat(k, :) = mvaf_single;
        
            mrse_lin_run_50 = [mrse_lin_run_50; a{end,end}];
        end
%         
%         disp('Mean mse')
%         disp(mean(mse_lin_mat)*100)
%         disp(mean(mean(mse_lin_mat))*100)
%         disp('Mean R^2')
%         disp(mean(mvaf_lin_mat)*100)
%         disp(mean(mean(mvaf_lin_mat))*100)
        
        mrse_lin_run_100 = [];
        mes_single_pred_run = zeros(size(states,1)/length_traj, 4);
        for k = 1:size(states,1)/length_traj-1
            a = information_criteria(states(1+(k-1)*length_traj:(k-1)*length_traj + length_traj, 1:length(eigenvalues_pred)), ...
                states_pred(1+(k-1)*length_traj:(k-1)*length_traj + length_traj,1:length(eigenvalues_pred)), length(eigenvalues_pred));
            mes_single_pred_run(k,:) = cell2mat(a(2,1:4));
            mrse_lin_run_100 = [mrse_lin_run_100; a{end,end}];
        end

        mrse_lin_runs_50 = [mrse_lin_runs_50; mean(mrse_lin_run_50)*100];
        mrse_lin_runs_100 = [mrse_lin_runs_100; mean(mrse_lin_run_100)*100];

        measures_lin_single_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(mes_lin_single_run, 1)), {length(eigenvalues_pred)}];
        measures_lin_single_pred_mean(end+1, :) = [{content(j + 2).name}, num2cell(mean(mes_single_pred_run, 1)), {length(eigenvalues_pred)}];
        
        % % 
        
     end
end

% save(['eval_2_outcome_full_p' num2str(length_traj) '.mat'])

figure(fig_eigs)
grid on;
plot(real(eig_ref), imag(eig_ref), 'bo')
plot(circle(1, :),circle(2, :), 'k-')
xlabel('Realteil')
ylabel('Imaginärteil')
set(gca, 'xLim', [-0.5; 1.5])
set(gca, 'yLim', [-1; 1])
% set(gca,'DataAspectRatio',[1 1 1])  
leg{end+1} = 'Linearization';
leg{end+1} = 'Einheitskreis';
legend(leg)






            

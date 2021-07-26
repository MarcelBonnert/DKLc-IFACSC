% state space
data_full = cell2mat(measures_single_mean(2:end,2:end));
data_full = cell2mat(measures_single_pred_mean(2:end,2:end));

% Koopman space
% data_full = cell2mat(measures_lin_single_mean(2:end,2:end));
% data_full = cell2mat(measures_lin_single_pred_mean(2:end,2:end));

%% Added states
for i = 1:1
    data = data_full(:, 1 + (i-1)*5:i*5);
    xlabel = data(:, end);
    
    sortarray = [3:8, 1:2, [3:8, 1:2]+8, [3:9, 1:2]+16, [3:9, 1:2]+25];
    
    xlabel = data(sortarray, end);

    figure
    
    hold on
    plot(1:size(data, 1), data(sortarray, 2), 'x')
    
    patch([0.5 8.5 8.5 0.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.8941 0.9412 0.9020])
    patch([8.5 16.5 16.5 8.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.8706 0.9216 0.9804])
    patch([16.5 25.5 25.5 16.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.9373 0.8667 0.8667])
    patch([25.5 34.5 34.5 25.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.9922 0.9176 0.7961])
    xlim([0.5 34.5])
    ylim([min(min(data(sortarray,1:3)))*1.02 max(max(data(sortarray,1:3)))*0.98])

    plot(1:size(data, 1), data(sortarray, 2), 'x', 'color', [0 0.4471 0.7412]) % [0.8510 0.3255 0.0980]

    txt = '(1)';
    text(4, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(2)';
    text(12, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(3)';
    text(20.5, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(4)';
    text(29.5, max(max(data(sortarray,1:3)))*0.995,txt)

    ax = gca;
    ax.XTick = [1:2:size(data, 1)];
    ax.XTickLabel = num2cell([xlabel(1:2:end)]);
end

%% plain state
for i = 1:1
    data = data_full(:, 1 + (i-1)*5:i*5);
    xlabel = data(:, end);
    
    sortarray = [3:9, 1:2, [3:9, 1:2]+9, [2 4:10 1 3]+18, [2 4:10 1 3]+28];
    xlabel = xlabel(sortarray);

    figure
    hold on
    plot(1:size(data, 1), data(sortarray, 2), 'x')

    patch([0.5 9.5 9.5 0.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.8941 0.9412 0.9020])
    patch([9.5 18.5 18.5 9.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.8706 0.9216 0.9804])
    patch([18.5 28.5 28.5 18.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.9373 0.8667 0.8667])
    patch([28.5 38.5 38.5 28.5], [max(max(data(sortarray,1:3)))*0.98 ...
        max(max(data(sortarray,1:3)))*0.98 ...
        min(min(data(sortarray,1:3)))*1.02 ...
        min(min(data(sortarray,1:3)))*1.02], [0.9922 0.9176 0.7961])
    xlim([0.5 38.5])
    ylim([min(min(data(sortarray,1:3)))*1.02 max(max(data(sortarray,1:3)))*0.98])

    plot(1:size(data, 1), data(sortarray, 2), 'x', 'color', [0 0.4471 0.7412]) % [0.8510 0.3255 0.0980]

    txt = '(1)';
    text(4, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(2)';
    text(13.5, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(3)';
    text(23, max(max(data(sortarray,1:3)))*0.995,txt)
    txt = '(4)';
    text(33, max(max(data(sortarray,1:3)))*0.995,txt)

    ax = gca;
    ax.XTick = [1:2:size(data, 1)];
    ax.XTickLabel = num2cell([xlabel(1:2:end)]);

end


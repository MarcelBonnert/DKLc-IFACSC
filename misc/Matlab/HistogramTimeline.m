classdef HistogramTimeline < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(GetAccess = private)
        list;                           %list of classes
        timelinehisto;
        listSize;
        timeSize;
        scale_y;
    end
        
        
    
    methods
        function obj = HistogramTimeline(list_t, timeSize_t, scale_y_t) %timeSize= how many timesteps
            obj.list = list_t;
            s = size(list_t);
            s = s(2);
            obj.listSize = s;
            obj.timeSize = timeSize_t;
            obj.timelinehisto = zeros(s-1, timeSize_t);
            obj.scale_y = scale_y_t;
        end
        
        function A = sub(this, histo)
            A = this.timelinehisto-histo.getTimeLineHisto();
        end
        
        function tLH = getTimeLineHisto(this)
            tLH = this.timelinehisto;
        end
        
        function addData(this, data, k)                       %k = time, data = datapoint
            for index = (1:this.listSize-1)
                if(data<this.list(index+1) && data>=this.list(index))
                    this.timelinehisto(index,k) = this.timelinehisto(index,k)+1;
                    break;
                end
            end
            if data >= this.list(this.listSize)
                this.timelinehisto(this.listSize-1,k) = this.timelinehisto(this.listSize-1,k)+1;
            end
            if data <= this.list(1)
                this.timelinehisto(1,k) = this.timelinehisto(1,k)+1;
            end
        end
        
        function plot(this, line_at)  
            imagesc((0:this.timeSize), this.list(2:end), this.timelinehisto / this.scale_y);
            C = jet();
            colormap(C);
            colorbar();
            set(gca,'ColorScale','log')
            
            [mu_prog, var_prog] = this.getMeanVar();
            hold on
            plot(0:this.timeSize-1, mu_prog, '--k', 'LineWidth', 2)
            plot(0:this.timeSize-1, mu_prog + (var_prog) * 3, '-.m', 'LineWidth', 2)
            plot(0:this.timeSize-1, mu_prog - (var_prog) * 3, '-.m', 'LineWidth', 2)
            ylim([this.list(1) this.list(end)])
            
            if nargin > 1
                plot(ones(2,1)*line_at, [this.list(1) this.list(end)], 'y:', 'LineWidth', 2)
            end
        end
        
        function [mu_prog, var_prog] = getMeanVar(this)
            e = this.list(2:end);
            mu_prog = zeros(size(this.timelinehisto, 2), 1);
            var_prog = zeros(size(this.timelinehisto, 2), 1);
            
            for i = 1:size(this.timelinehisto, 2)
                vec = [];
                for j = 1:size(this.timelinehisto, 1)
                    vec = [vec; ones(this.timelinehisto(j, i),1) * e(j)];
                end
                [mu_prog(i) ,var_prog(i)] = normfit(vec);
            end
            
        end
        
        function plot3d(this, kind)  
            if nargin < 2
                kind = 'abs';
            end
            xx = linspace(this.list(1), this.list(end), length(this.list));
            Z = exp(-(xx'-mu_prog').^2./var_prog');
            maxVal = max(this.timelinehisto);
            switch kind
                case 'abs'
                    plotData = this.timelinehisto;
                    Z = Z .* maxVal;
                case 'rel'
                    plotData = this.timelinehisto./maxVal;
            end
               
            subplot(2,1,1)
            bar3(plotData)
            
            fig = gcf;
            for i = 1:length(fig.Children.Children)
%                 dist = 
                fig.Children.Children(i).EdgeAlpha = 0;
            end
            ylim([0, this.listSize-1])
            fig.Children.YTickLabel = num2cell(linspace(this.list(1), this.list(end), length(fig.Children.YTickLabel)));
            
            [mu_prog, var_prog] = this.getMeanVar();
            ax2 = subplot(2,1,2);
            
            [X, Y] = meshgrid(this.list, linspace(0,this.timeSize, this.timeSize));
            surf(X', Y', Z, 'EdgeAlpha', 0)
            ax2.View = [32, 19];
        end
        
        function save(this, file_name)
            for ind1 = (1:this.timeSize)
                out = zeros((this.listSize-1), 3);
                for ind2 = (0:this.listSize-2)
                    out(1 + ind2, 1) = (ind1-1);
                    out(1 + ind2, 2) = this.list(ind2+1);
                    temp = this.timelinehisto(ind2+1, ind1);
                    if temp == 0
                        temp = 0.1;
                    end
                    out(1 + ind2, 3) = temp;
                end
                if ind1 == 1
                    dlmwrite(file_name, out, 'delimiter', ' ');
                else
                    fid=fopen(file_name,'at+');
                    fprintf(fid,'\n');
                    fclose(fid);
                    dlmwrite(file_name, out, 'delimiter', ' ', '-append');
                end
            end
        end
                   
    end
    
end


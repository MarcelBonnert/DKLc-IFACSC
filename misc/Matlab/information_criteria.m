function measures = information_criteria(y, y_hat, Np)
% First bracket - Denominator order; second bracket - numerator order
% % third bracket - delay
N = size(y,1);
epsilon = mean(diag((y_hat-y)'*(y_hat-y)))/N;

eta_aic = 2*Np; 
% AICC differs from source to source
eta_aicc = 2*N*Np/(N-Np-1);
% Can be varied from 3 to 6 
eta_bic = log(N)*Np;

logsig = log(epsilon);
% Stoica, O., Selén, Y. 2004 - Model-Order Selection: A review of
% information criterion rules
% N = 1;
measures(1) = N * logsig+eta_aic;
measures(2) = N * logsig+eta_aicc;
measures(3) = N * logsig+eta_bic; % Same as MDL
measures(4) = epsilon;
measures(5) = mrse(y, y_hat);
measures = [{'AIC' 'AIC_c' 'BIC' 'eps' 'MRSE'};num2cell(measures)];
end

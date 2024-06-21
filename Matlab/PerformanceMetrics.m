function [PearCorr, RMSE, SNR] = PerformanceMetrics(EEG1, EEG2)
% [PearCorr, RMSE, SNR] = PerformanceMetrics(EEG1, EEG2)
%
% Inputs:
%    EEG1           is a multi-channel time-series (samples by channels)
%                   of the blink-free ground truth EEG set.
%    EEG2           is a multi-channel time-series (samples by channels)
%                   of the blink-cleaned using a blink-removal method 
%                   EEG set.
%
% Outputs:
%    PearCorr       is a vector of the same size as the number of channels 
%                   that contains the Pearson correlation between EEG1 and 
%                   EEG2.
%    RMSE           is a vector of the same size as the number of channels 
%                   that contains the Root Mean Square Error between EEG1  
%                   and EEG2.
%    SNR            is a vector of the same size as the number of channels 
%                   that contains the Signal-to-Noise ratio between EEG1  
%                   and EEG2.

if nargin < 2
    error('At least two input arguments are required.');
end

PearCorr = [];
RMSE     = [];
SNR      = [];

s1 = size(EEG1);
if s1(1) < s1(2)
    EEG1 = EEG1';
end

s2 = size(EEG2);
if s2(1) < s2(2)
    EEG2 = EEG2';
end

for chn = 1:size(EEG1, 2)
    RMSE     = [RMSE, sqrt(mean((EEG1(:,chn) - EEG2(:,chn)).^2))];
    PearCorr = [PearCorr   , corr2(EEG1(:,chn), EEG2(:,chn))];
    SNR      = [SNR     , 10*log10(std(EEG1(:,chn)) / std(EEG1(:,chn) - EEG2(:,chn)))];

end


end
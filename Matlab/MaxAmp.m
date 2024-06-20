function [max_amp] = MaxAmp(data_vector, fs, window_size, shift_size)
% [max_amp] = MaxAmp(data_vector, fs, window_size, shift_size)
%
% Inputs:
%    data_vector    is a time-series of N samples
%    fs             is the sampling frequency in Hz
%    window_size    is the size of data to be processed 
%    shift_size     is the shift amout to get to the next window
%
% Outputs:
%    max_amp        is a cell of of `M` values corresponsing to the 
%                   absolute maximum of the window_size second long data 
%                   segments


if nargin < 4
    shift_size = 15;  % Default window size is 15 seconds
end
if nargin < 3
    window_size = 15; % Default shift is 15 seconds
end
if nargin < 2
    error('At least two input arguments are required.');
end

window_size_pts = floor(window_size * fs);
shift_size_pts  = floor(shift_size  * fs);

max_amp = [];

for i = 1:shift_size_pts+1:length(data_vector)
    
    start  = i;
    finish = min([i + window_size_pts, length(data_vector)]);

    window_data = data_vector(start:finish,:);
    max_amp = [max_amp max(abs(window_data))];
end


end
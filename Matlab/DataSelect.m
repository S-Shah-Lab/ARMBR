function [filtered_data, filtered_data_inx, excluded_points, excluded_points_inx] = DataSelect(data_vector, init_size, std_dev_threshold)
% [filtered_data, filtered_data_pts, excluded_points, excluded_points_pts] = DataSelect(data_vector, init_size, std_dev_threshold)
%
% Inputs:
%    data_vector        is a vector of N values from which we seek to eliminate
%                       outliers
%    init_size          is the number of data points from `data_vector` used
%                       for initialization 
%    std_dev_threshold  is the threshold over which points are considered
%                       outliers
%
% Outputs:
%    filtered_data          is a vector of values excluding the outliers
%    filtered_data_inx      is a vector of indices of the values excluding 
%                           the outliers
%    excluded_points        is a vector of the excluded the outliers
%    excluded_points_inx    is a vector of indices of the excluded outliers



if nargin < 3
    std_dev_threshold = 5;  % Default threshold size is 5 
end
if nargin < 2
    init_size = 3; % Default initial_size is 3 points
end
if nargin < 1
    error('At least one input arguments are required.');
end


% Empty arrays to store filtered data
filtered_data = [];
filtered_data_inx = [];

excluded_points = [];
excluded_points_inx = [];

data_vector_ = data_vector;


% Iterate through the data
for j = 1:min([init_size, length(data_vector)])
    filtered_data = [filtered_data {data_vector_(j)}];
    filtered_data_inx = [filtered_data_inx j];
end



for i = init_size+1:length(data_vector_)

    % Extract the previous points
    previous_points = data_vector_(1:i-1);

    % Calculate mean and standard deviation of the previous 3 points
    mean_prev = mean(previous_points);
    std_dev_prev = std(previous_points);

      
    % Check if the current point is within 2 standard deviations
    if abs(data_vector_(i) - mean_prev) <= std_dev_threshold * std_dev_prev
        
        % Include the current point in the filtered data
        filtered_data = [filtered_data {data_vector_(i)}];
        filtered_data_inx = [filtered_data_inx i];

    else
        % Exclude the current point
        data_vector_(i) = mean_prev;
        excluded_points = [excluded_points {data_vector_(i)}];
        excluded_points_inx = [excluded_points_inx i];
    end
end



end





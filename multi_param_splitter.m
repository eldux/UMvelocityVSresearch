function output = multi_param_splitter(input, n, s)
%MULTI_PARAM_SPLITTER Summary of this function goes here
%   Detailed explanation goes here
%   input - is multiparameter time series
%   n - window size
%   s - step size
%   output - array of cells of multiparameter windows
len_of_data = size(input,2);
sampling = 1:s:(len_of_data-n);
output = cell(length(sampling), 1); % Preallocating cell array
for i = sampling
    output{i} = input(:, i:(i+n-1));
end
end


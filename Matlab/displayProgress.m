function displayProgress(progress, totalSteps)
    percent = progress / totalSteps * 100;
    numSymbols = floor(percent / 2);  % Each symbol represents 2%
    
    % Build progress bar
    progressBar = ['#' repmat('#', 1, numSymbols) sprintf(' %.1f%%', percent)];
    
    % Display progress bar
    fprintf('%s\r', progressBar);
    
    % Clear previous line if progress is complete
    if progress == totalSteps
        fprintf('\n');
    end
end



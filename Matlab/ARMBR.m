function [no_blink_eeg, Opt_Alpha, Blink_Ref, Blink_Artifact, BlinkSpatialPattern] = ARMBR(orig_eeg, blink_chan_nbr, fs, Alpha)
% [no_blink_eeg, Opt_Alpha, Blink_Ref, Blink_Artifact] = ARMBR(orig_eeg, blink_chan_nbr, fs, Alpha)
%
% Inputs:
%    orig_eeg       is a multi-channel time-series (samples by channels)
%    blink_chan_nbr is a vector of indices corresponding to the channels
%                   the most affected by blinks (EOG or Fp1,Fp2)
%    fs             is the sampling frequency in Hz
%    Alpha          is the blink level threshold. Alpha can be manually
%                   entered. If set to -1, an automatic search is performed. 
%
% Outputs:
%    no_blink_eeg   multi-channel time-series after blinks are suppressed.
%    Opt_Alpha      is the optimal Alpha found by the automatic search
%                   algorithm.
%    Blink_Artifact is a time-series of the blink component that is removed
%                   from the eeg data matrix.
%    Blink_Ref      is a train of pulses at the blink location.


if nargin < 4
    Alpha = -1;  % Default value Alpha is -1 which for automatic selection
end
if nargin < 3
    error('At least three input arguments are required.');
end


% Ensure that EEG is in the format (samples x channels)
% We assume that there are more samples than channels
if size(orig_eeg, 1) < size(orig_eeg, 2)
    orig_eeg = orig_eeg';
end

% Segment the EEG data into smaller segments and remove bad segments
[good_eeg, ~, good_blinks] =  Data_Prep(orig_eeg, fs, blink_chan_nbr);


if Alpha == -1 % Run the automatic Alpha selection mechanism
    Delta = [];
    alpha_range = 0.01:0.1:10;

    for alpha = alpha_range
        displayProgress(alpha, 10);
        [~, Blink_Artifact, ~, ~] = Blink_Selection(orig_eeg, good_eeg, good_blinks, alpha);
                                                           
        if ~sum(isnan(Blink_Artifact))
            LPF = bandpass(Blink_Artifact,  [1 8],  fs, 'Steepness', 0.99);
            Delta = [Delta sum(LPF.^2)/(sum((Blink_Artifact - LPF).^2))];
        else
            break
        end
    end
    
    % Remove NaN values
    alpha_range = alpha_range(~isnan(Delta)); Delta = Delta(~isnan(Delta)); 
    

    % Find optimal Alpha
    if ~isempty(Delta)
        [~, Delta_Max_inx] = max(Delta);
        Opt_Alpha = alpha_range(Delta_Max_inx);
         
        [no_blink_eeg, Blink_Artifact, Blink_Ref, BlinkSpatialPattern] = Blink_Selection(orig_eeg, good_eeg, good_blinks, Opt_Alpha);
                
    else
        no_blink_eeg = orig_eeg;
        Blink_Artifact = [];
        Blink_Ref = [];
        Opt_Alpha = [];

    end

else % if Alpha is manually entered

    [no_blink_eeg, Blink_Artifact, Blink_Ref, BlinkSpatialPattern] = Blink_Selection(orig_eeg, good_eeg, good_blinks, Alpha);
    Opt_Alpha = Alpha;

end

end
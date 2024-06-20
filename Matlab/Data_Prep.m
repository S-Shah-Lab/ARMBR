function [GoodEEG, OrigEEG, GoodBlinks] =  Data_Prep(eeg, fs, blink_chan_nbr)
% [eeggood_segs, eeg_orig, Blinkgood_segs] =  Data_Prep(eeg, fs, blink_chan_nbr)
%
% Inputs:
%    eeg            is a multi-channel time-series (samples by channels)
%    fs             is the sampling frequency in Hz
%    blink_chan_nbr is a vector of indices corresponding to the channels
%                   the most affected by blinks (EOG or Fp1,Fp2)
%
% Outputs:
%    GoodEEG    is a multi-channel time-series of `good` EEG signals.
%    OrigEEG    is the same multi-channel time-series as the input `eeg`
%    GoodBlinks is a time-series of `good` blink signal.


if nargin < 3
    error('At least three input arguments are required.');
end


% Ensure that EEG is in the format (samples x channels)
% We assume that there are more samples than channels
if size(eeg, 1) < size(eeg, 2)
    eeg = eeg';
end
OrigEEG = eeg;

% Combine all blink reference channels in one vector
Blink = mean(eeg(:, blink_chan_nbr), 2);

% Check if skewed left. If so, make it skewed right.
if median(Blink) > mean(Blink)
    Blink = -Blink;
end


% Find max value during each `window_size` (Default is 15 sec) second segments.
[BlinkAmp]     = MaxAmp(diff(Blink), fs);

% Segment the data into `window_size` (Default is 15 sec) second segments.
[Blink_epochs] = Segment(Blink,      fs);
[eeg_epochs]   = Segment(eeg,        fs);


% Select only good data segments to train ARMBR on
[~, filtered_data_pts, ~, ~] = DataSelect(BlinkAmp);


GoodBlinks = [];
GoodEEG = [];

for g = 1:length(Blink_epochs)

    if sum(filtered_data_pts == g)
        GoodBlinks = [GoodBlinks; Blink_epochs{g}];
        GoodEEG    = [GoodEEG;   eeg_epochs{g}];

    end
end





end


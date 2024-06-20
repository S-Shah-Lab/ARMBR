function [no_blink_eeg, Blink_Artifact, ref] = Blink_Selection(eeg_orig, eeg_filt, Blink_filt, alpha0, maskIn)
% [no_blink_eeg, Blink_Artifact, ref] = Blink_Selection(eeg_orig, eeg_filt, Blink_filt, alpha0, maskIn)
%
% Inputs:
%    eeg_orig       is a multi-channel time-series (samples by channels)
%                   that typically contains all EEG data 
%    eeg_filt       is a multi-channel time-series (samples by channels)
%                   that typically contains a reduced set of the original
%                   EEG. `eeg_filt` exludes bad segments and is used to
%                   train ARMBR weights.
%    Blink_filt     is a time-series that typically contains a reduced set 
%                   of the blink reference signal. `Blink_filt` exludes bad
%                   segments and is used to train ARMBR weight

%    Alpha0         is the optimal blink level threshold.
%    maskIn         is a vector of the same size as the number of channels.
%                   Use `1` for the indices of the channels that will be
%                   included in the analysis and `0` for those that are
%                   excluded from the analysis.
%
% Outputs:
%    no_blink_eeg   multi-channel time-series after blinks are suppressed.
%    Blink_Artifact is a time-series of the blink component that is removed
%                   from the eeg data matrix.
%    Blink_Ref      is a train of pulses at the blink location.


if nargin < 5
    nChannels = size(eeg_orig,2);
    maskIn = ones(nChannels, 1); % all channels are used. 
end
if nargin < 4
    error('At least four input arguments are required.');
end




% Compute the Inter-quartile Range
Qa = quantile(Blink_filt, 0.159);
Qb = quantile(Blink_filt, 0.841);
Q2 = quantile(Blink_filt, 0.5);


StD = (Qb-Qa)/2;
T0 = Q2+alpha0*StD;


% Build a reference signal, which is a train of 1 and 0, where 1 indicates a blink and 0 indicates no blink
eeg_reduced   = eeg_filt(Blink_filt > Qa, :);
Blink_reduced = Blink_filt(Blink_filt > Qa);
ref           = Blink_reduced > T0;


% Project out blink components
if sum(ref) ~= 0
    [~, ~, ~, ~, Blink_Artifact, no_blink_eeg] = projectout(eeg_orig, eeg_reduced, ref, maskIn);
else
    Blink_Artifact = [];
    no_blink_eeg = [];
end

end


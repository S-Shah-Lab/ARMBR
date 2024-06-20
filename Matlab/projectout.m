function [M_purge, w, a, Sigma, S, X_purged] = projectout(X, X_reduced, ref, maskIn)

% [M_PURGE, W, A, SIGMA] = projectout(X, X_reduced, REF[, MASK_IN])
%
% Inputs:
%    X         is a multi-channel time-series (samples by channels)
%    X_reduced is a multi-channel time-series (samples by channels), with
%              the data that will be used to train ARMBR
%    REF       is a reference signal (samples by n,  typically n=1)
%    MASK_IN   is an optional logical matrix dictating which of X's columns to consider
%              (e.g. indicating which channels are EEG channels as opposed to triggers, etc)
% Outputs:
%    M_PURGE   is a square spatial filtering matrix (each COLUMN is a spatial filter) for
%              removing correlates of REF from X  (X_PURGED = X * M_PURGE)
%    W         is a spatial filter for estimating the signal to be eliminated
%    A         is a spatial pattern indicating where the signal projects to
%    SIGMA     is a covariance matrix of the data (but only in rows & columns where MASK_IN
%              is true---other rows/columns are equal to the corresponding rows/columns
%              of the identity matrix)
%    S         is the signal corresponding to each column of A
%    X_PURGED  is the signal X, in the original domain, after S has been projected out of it
%
% One good use of this function is to remove blinks from EEG.  In this case, REF could be
% a series of rectangular pulses marking the blinks (or even deltas marking their peaks).
% For example, you could prepare REF as follows:
%
%     ref = eeg(:, indices_of_channels_adjacent_to_eyes);
%     ref = mean(ref, 2);            % mean across channels
%     ref = high_pass_filter(ref);   % use your favourite high-pass-filter implementation here BUT MAKE SURE IT DOES NOT INTRODUCE A GROUP DELAY
%     ref = abs(ref);                %
%     ref = (ref > some_threshold);  % determine the threshold by eye after plotting, or use a quantile-based method
%
% Another nice option is to use REF=[cos(2*pi*60*t_sec) sin(2*pi*60*t_sec)] to remove
% 60Hz power line interference.

[nSamples, nChannels] = size(X_reduced);
nRefs = size(ref, 2);

if nargin < 5, maskIn = []; end
if isempty(maskIn), maskIn = logical(ones(nChannels, 1)); end
maskIn = maskIn(:);
if ~islogical(maskIn)
    if min(maskIn)>0 & (max(maskIn)>1 | numel(maskIn) ~= nChannels), ind = maskIn; maskIn = zeros(nChannels, 1); maskIn(ind) = 1; end
    maskIn = logical(maskIn);
end
if numel(maskIn) ~= nChannels, error(sprintf('the number of channels implied by MASK_IN (%d) does not match the number of columns in X (%d)', numel(maskIn), nChannels)), end
maskOut = ~maskIn;

if size(ref, 1) ~= nSamples, error(sprintf('the number of rows in REF (%d) does not match the number of rows in X (%d)', size(ref, 1), nSamples)), end



I = eye(nChannels);
Sigma = cov(X_reduced);
Sigma(:, maskOut) = I(:, maskOut);

Sigma(maskOut, :) = I(maskOut, :);

if ~isa(X_reduced, 'double'), ref = double(ref); end
wMasked = [X_reduced(:, maskIn) ones(nSamples, 1)] \ ref;

bias = wMasked(end, :);
w = zeros(nChannels, nRefs); 
w(maskIn, :) = wMasked(1:end-1, :);
v = sum( ( Sigma * w ) .* w, 1 );

rescale = diag(v.^-0.5);
w = w * rescale;


a = Sigma * w;
M_est = w * a';


M_purge = I - M_est;
if nargout >= 5, S = X * w; end
if nargout >= 6, X_purged = X * M_purge; end

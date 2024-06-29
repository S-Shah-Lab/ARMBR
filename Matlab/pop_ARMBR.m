function [EEG] = pop_ARMBR(EEG, blink_chan)
% Inputs:
% EEG       - EEGLAB data structure.
%
% Outputs:
% EEG       - EEGLAB data structure


EEG_ = EEG.data;
Fs_  = EEG.srate;


% Check if indices or channel names are used
allStrings = isstring(blink_chan);
if allStrings
    channel_names = {EEG.chanlocs.labels};
    channel_names_ = [];
    for i = 1:length(channel_names)
        channel_names_ = [channel_names_ lower(string(channel_names{i}))];
    end
    blink_chan_ = [];
    for j = 1:length(blink_chan)
        indices = find(channel_names_ == lower(blink_chan(j)) );
        blink_chan_ = [blink_chan_ indices];
    end 
    blink_chan = blink_chan_;


else
    blink_chan = blink_chan;
end


[ARMBR_EEG, ~, ~, ~, BlinkSpatialPattern] = ARMBR(EEG_, blink_chan, Fs_);

EEG_size = size(ARMBR_EEG);
if EEG_size(1) > EEG_size(2)
    ARMBR_EEG = ARMBR_EEG';
end
EEG.data = ARMBR_EEG;
EEG.BlinkSpatialPattern = BlinkSpatialPattern;

disp('EEG is updated!')

end
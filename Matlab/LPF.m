function [LPFeegData] = LPF(eegData, fs, Freq_Band)

        hd_lpf = getLPFilt(fs,Freq_Band);
        temp_lpf_EEG = cellfun(@(x) filtfilthd(hd_lpf,x),{eegData},'UniformOutput',false);
        LPFeegData = temp_lpf_EEG{:};


end
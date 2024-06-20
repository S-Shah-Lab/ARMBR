function [BPFeegData] = BPF(eegData, fs, Freq_Bands)


        hd_lpf = getLPFilt(fs,Freq_Bands(2));
        temp_lpf_EEG = cellfun(@(x) filtfilthd(hd_lpf,x),{eegData},'UniformOutput',false);
        LPFeegData = temp_lpf_EEG{:};

        hd_hpf = getHPFilt(fs,Freq_Bands(1));
        temp_hpf_EEG = cellfun(@(x) filtfilthd(hd_hpf,x),{LPFeegData},'UniformOutput',false);
        BPFeegData = temp_hpf_EEG{:};

end
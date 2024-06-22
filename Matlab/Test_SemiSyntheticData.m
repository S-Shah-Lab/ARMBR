clc
close all
clear

fs = 128; % Set sampling rate
sub = 1;  % Set subject number

% Load clean and blink-contaminated EEG signals
Clean_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Clean_EEG.mat").Clean_EEG;
Sythentic_Blink_Contaminated_EEG = load("..\SemiSyntheticData\Sub"+num2str(sub)+"\Sub"+num2str(sub)+"_Synthetic_Blink_Contaminated_EEG.mat").Sythentic_Blink_Contaminated_EEG;

% Bandpass filter the data from 1 to 40
Sythentic_Blink_Contaminated_EEG = BPF(Sythentic_Blink_Contaminated_EEG, fs, [1 40]);
Clean_EEG                        = BPF(Clean_EEG, fs, [1 40]);

% Run ARMBR
ref_chan_nbr = [80, 93]; %indices for Fp1 and Fp2
[ARMBR_EEG, Set_IQR_Thresh, Blink_Ref, Blink_Artifact, BlinkSpatialPattern] = ARMBR(Sythentic_Blink_Contaminated_EEG, ref_chan_nbr, fs);

% Compute performance metrics
[PearCorr, RMSE, SNR] = PerformanceMetrics(Clean_EEG, ARMBR_EEG);

% Display computed metrics
disp(['========================================='])
disp(['Pearson correlation for subject ',num2str(sub),': ', num2str(round(mean(PearCorr), 2))])
disp(['SNR                 for subject ',num2str(sub),': ', num2str(round(mean(SNR), 2))])
disp(['RMSE                for subject ',num2str(sub),': ', num2str(round(mean(RMSE), 2))])
disp(['========================================='])





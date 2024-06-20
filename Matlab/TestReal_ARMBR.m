clear
clc
close all

main_path = 'D:\Users\ludvi\Dropbox (Personal)\Cornell Postdoc\Codes and Tools\Shah Lab Toolbox\Matlab\Blink Removal with Hill method\EEG Real Data - Mastoids\';
addpath('D:\Users\ludvi\Dropbox (Personal)\Cornell Postdoc\Codes and Tools\Shah Lab Toolbox\Matlab\Blink Removal with Hill method\ARMBR V02\Matlab\')
folder = {dir(main_path).name};
folder = folder(3+18:end-1);

ref_chan_nbr = [80, 93];



%%
for f = 2% 1:length(folder)

    runs = {dir([main_path,folder{f}]).name};
    runs = runs(3:end-1);

    for r = 1:length(runs)

        runs{r}


        EEG1 = load([main_path,folder{f},'/',runs{r}]);

        EEG1.eegData = EEG1.eegData - mean(EEG1.mastoids, 2);


        [Orig_EEG] = ARMBR_BPF(EEG1, [1 40]);

        tic
        [ARMBR_EEG, Set_IQR_Thresh, Blink_Ref, Blink_Artifact] = ARMBR(Orig_EEG.eegData, ref_chan_nbr, -1, EEG1.fs);
        toc

        save([main_path,folder{f},'/cleaned/ARMBR_',runs{r}], 'ARMBR_EEG')



        mv = 100;

        figure
        for i = 1:128
            plot(Orig_EEG.eegData(:,i) + i*mv, 'r')
            hold on
            plot(ARMBR_EEG(:,i)+ i*mv, 'k')
        end
        hold off
        title(runs{r})

    end


end



%%


figure
mv = 100;
for i = 1:128
    plot(Orig_EEG1.eegData(:,i) + i*mv, 'r')
    hold on
    plot(ARMBR_EEG(:,i)+ i*mv, 'k')
end
hold off

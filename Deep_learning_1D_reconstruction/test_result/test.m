%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% test result
% 12/18/2022
%
clear all;clc
load prediction.mat
load test_label.mat
load waven.mat

k=1;
for i=1:488
    m=1;
    for n=1:600
        if prediction(i,n)~=0
           wave_new(m)=waven(n);
           result(m)=prediction(i,n);
           m=m+1;
        end
    end

    figure
    plot(wave_new, result,'LineWidth',3);
    hold on
    plot(waven,test_label(i,:),'r:','LineWidth',3);

end
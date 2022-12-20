%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for 2D pattern
% 12/18/2022
%

clear all; clc;
load test_label.mat
load prediction_v.mat
load prediction_h.mat
load waven.mat


for i=13:24
    m=1;
    for n=1:440
        if prediction_v(i,n)~=0;
           wave_new(m)=waven(n);
           result(m)=prediction_v(i,n);
           m=m+1;
        end
    end
    cos_value=max(result)-result(1);
    figure
    subplot(1,2,1)
    plot(wave_new, result,waven,test_label(i,1:440));
    clear wave_new; clear result;
    m=1;
    for n=1:440
        if prediction_h(i,n)~=0;
           wave_new(m)=waven(n);
           result(m)=prediction_h(i,n);
           m=m+1;
        end
    end
    sin_value=max(result)-result(1);
    subplot(1,2,2)
    plot(wave_new, result,waven,test_label(i,441:880));
    deg(i)=atan(sin_value/cos_value)*180/pi;
end


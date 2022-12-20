%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% ORD & polarization
% 12/18/2022
%

clear all;
load prediction.mat


[x,y,z]=size(prediction);
w=[470, 490, 500, 525, 550, 595, 635, 660, 740];
for i=1:x
    temp=[1:1:30];
    predict=squeeze(prediction(i,:,:));
    result(i,:)=temp*predict';
end
% if the reuslt is 30 indicated it not belong to this wavelength
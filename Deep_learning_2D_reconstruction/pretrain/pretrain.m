%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for 2D pattern
% 12/18/2022
%
clear all; clc;
load label.mat %load spectrum
load tiffdata.mat; %load 2D image
load degree.mat;%polization for each image
load wavelength.mat

label=label';
% get rid of useless information
label(:,1402:2048)=[];
label(:,1:81)=[];
%reduce length
k=3;
for i=1:1320/k
    all_label(:,i)=label(:,i*k);
end
[x,y]=size(all_label);
for i=1:x
    noise=all_label(i,1);
    temp=all_label(i,:)-noise;
    degree=degree*pi/180;
    all_label1(i,:)=temp.*cos(degree)+noise;
    all_label2(i,:)=temp.*sin(degree)+noise;
end


all_label=cat(2,all_label1,all_label2);

%% get train data from all_label tiffdata
% train_data=...
% train_label=...
%% get train data from all_label tiffdata
% test_data=...
% test_label=...

train_data = permute(train_data,[3 1 2]);
test_data = permute(test_data,[3 1 2]);

save('train_label.mat','train_label');
save('train_data.mat','train_data');
save('test_label.mat','test_label');
save('test_data.mat','test_data');
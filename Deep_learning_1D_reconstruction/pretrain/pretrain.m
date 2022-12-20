%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for 1D pattern
% 12/18/2022
%
clear all; clc;
% load data
load label.mat %load spectrum
load tiffdata.mat %load image
load wavelength.mat

label=label';
% get rid of useless information
label(:,1351:2048)=[];
label(:,1:150)=[];
wavelength=wavelength';
wavelength(:,1351:2048)=[];
wavelength(:,1:150)=[];
%reduce length
k=2;
for i=1:1200/k
    train(:,i)=label(:,i*k);
    waven(i)=wavelength(:,i*k);
end

% get train dataset


label=[45 12 30 18 30 36 36 18 18 36 12 12 6 6 12 30 30 18 18 18 18 8 8 8 8 8 8 8 8];

train_label=train(1:45,:);
train_data=tiffdata(:,:,1:45);

num=1;
for i=1:length(label)
    num=num+label(i);
    train_label((45+6*i-5):(45+6*i),:)=train(num:num+5,:);
    train_data(:,:,(45+6*i-5):(45+6*i))=tiffdata(:,:,num:num+5);
    
end
train_label(219:500,:)=train(253:534,:);
train_data(:,:,219:500)=tiffdata(:,:,253:534);

% get test dataset
test_label=train(47:534,:);
test_data=tiffdata(:,:,47:534);

train_data = permute(train_data,[3 1 2]);
test_data = permute(test_data,[3 1 2]);

save('train_label.mat','train_label');
save('train_data.mat','train_data');
save('test_label.mat','test_label');
save('test_data.mat','test_data');
save('waven.mat','waven');

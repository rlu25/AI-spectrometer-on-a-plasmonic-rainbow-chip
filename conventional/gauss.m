%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% conventional method
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
wavelength(1351:2048)=[];
wavelength(1:150)=[];
plot(wavelength,label(534,:));

% change 2D image into 1D vector
tiffdata=sum(tiffdata,2);
tiffdata=reshape(tiffdata,187,534);
tiffdata=tiffdata';
%reduce length
k=12;
for i=1:1200/k
    train(:,i)=label(:,i*k);
    waven(i)=wavelength(i*k);
end

% get train dataset
label=[45 12 30 18 30 36 36 18 18 36 12 12 6 6 12 30 30 18 18 18 18 8 8 8 8 8 8 8 8];
num=1;
for i=1:length(label)
    num=num+label(i);
    train_label(45+6*i-5:45+6*i,:)=train(num:num+5,:);
    train_data(45+6*i-5:45+6*i,:)=tiffdata(num:num+5,:);
end
% get test dataset
test_label=train(47:534,:);
test_data=tiffdata(47:534,:);

% get spectral response functions R 
for i=1:187
    cvx_begin
        variable x(100);
%         minimize(norm( T*x-test_data(:,i)));
        minimize( norm(train_label*x-train_data(:,i)) + 0.05 * norm(x,2) );
%         subject to
%             x >= 0;
    cvx_end
    R(:,i)=x;
end



R=R';
test_data=test_data';

% combination of Gaussian basis functions 
gau=zeros(100);
sigma=0.4247*2;
for i=1:100
    gau(:,i)=(sqrt(2*pi)*sigma).*exp(-(waven-waven(i)).^2./(2*sigma^2));
end
A=R*gau;

% fit the target spectrum
for i=1:488
    cvx_begin
        variable x(100);
%         minimize(norm( T*x-test_data(:,i)));
        minimize( norm(A*x-test_data(:,i),2 ) + 0.05 * norm(x,2) );
%         subject to
%             x >= 0;
    cvx_end
    alresult(:,i)=x;
end

alresult=alresult';
result=zeros(488,100);
% combination of Gaussian basis functions 
for j=1:488
    for i=1:100
        result(j,:)=result(j,:)+alresult(j,i).*gau(:,i)';
    end
end
alresult=result;
% use interpolation to enlarge the length
k=2;
for i=1:1200/k
    final_waven(i)=wavelength(i*k);
end
for i=1:488
    final_result(i,:)=interp1(waven,alresult(i,:),final_waven,'pchip');
end
save('final_result.mat','final_result');
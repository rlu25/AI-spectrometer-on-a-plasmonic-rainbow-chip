%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% 2D pattern reconstruction
% 12/18/2022


Requirment:

Software:
    Python == 3.11.0
    Tensorflow >= 1.0
    Matlab >= R2017b
Operating system:
    CPU or GPU
    Support Windows, Linux, Mac

Instructions to run code:

    1. Crop image into 2D pattern and save into tiffdata.mat, save spectrum into label.mat, save polarization of each image into degree.mat
    2. Run pretrain.m to get training/testing dataset
    3. Run cnn_ver.py and cnn_hor to train model, result save in the prediction_v.mat and prediction_h.mat
    4. Run test.m to plot the final result. (There have some zero point in result, we need to get rid of them)

Installing time around 1 hour (depend on operating system)
Running time around 12 hours (depend on number of training data)

Result have shown in Figure 4.
More detail please check supplementary information

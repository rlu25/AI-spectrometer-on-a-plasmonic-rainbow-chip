%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Resolution system
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

    1. Crop image into 2D pattern to get trainging or testing data, save spectrum as label (same as 1D/2D reconstruction).
    2. Get training/testing dataset
    3. Run cnn.py to train model, result save in the prediction.mat

Installing time around 1 hour (depend on operating system)
Running time around 3 hours (depend on number of training data)

Result have shown in Figure 3.
More detail please check supplementary information
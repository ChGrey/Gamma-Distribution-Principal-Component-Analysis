clear; close all; clc; 

%% Loading training set
trainset_name = '0_90';
load(strcat('./MSTARdata/kick/',trainset_name));

%% GPCA hyperparameters
GPCA.NumStages = 1;               % number of layers
GPCA.PatchSize = 17;           % kernel size
GPCA.NumFilters = 2;          % number of kernels
save_name = 'ks5';           % output name

TrnLabels = TRN(:,end);
X = TRN(:,:);
make_v (X,trainset_name,GPCA,save_name)
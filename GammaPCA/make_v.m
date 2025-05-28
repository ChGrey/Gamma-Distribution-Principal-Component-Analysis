function make_v (TRN,trainset_name,GPCA,save_name)
tic;
addpath('./Utils');
addpath('./Liblinear');
make; 
ImgFormat = 'gray'; %'color' or 'gray'
classes_indices = ["BMP2","BTR70","T72","2S1","BRDM2","D7","BTR60","T62","ZIL131","ZSU234"];

%% Loading data from training set
TrnData = TRN(:,1:end-1)';  
TrnData = TrnData/255;
clear TRN;

ImgSize = sqrt(size(TrnData,1)); 

% fprintf('\n ====== GPCA Parameters ======= \n')
% GPCA

%% GPCA Training
fprintf('\n ====== GPCA Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 
clear TrnData; 

[V, ~] = GPCA_train(TrnData_ImgCell,GPCA,1);

clear TrnData_ImgCell; 
V0 = cell2mat(V(1));

path = strcat('./kernels/',trainset_name,'/');
if ~exist(path)
    mkdir(path);
end

save_path = strcat(path,save_name);
save(save_path,'V0');
time = toc;
fprintf('\n used time: %.2f secs.\n',[time]);
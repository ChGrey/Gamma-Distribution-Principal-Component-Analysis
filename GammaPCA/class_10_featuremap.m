clc;clear;
%% 
i = 10;                                  % 图像类别
load('.\十类卷积核\0-90\V9.mat')        % 投影矩阵类别
%% 
class_name = '0-90';
load(strcat('./MSTARdata/kick/[1]TRN',class_name)); 
TrnLabels = TRN(:,end);
X = TRN(TrnLabels==i,:);
TrnData = (X(:,1:end-1)'-1);  % partition the data into training set and validation set
TrnData = (TrnData/255 - 0.5)/0.5;

PCANet.NumStages = 2;               % 2阶段
PCANet.PatchSize = [16 16];           % [k1 k2]
PCANet.NumFilters = [16 16];          % [L1 L2]
PCANet.HistBlockSize = [7 7];
PCANet.BlkOverLapRatio = 0.5;
PCANet.Pyramid = [];

ImgSize = 128;
ImgFormat = 'gray';
OutImg = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat);
ImgIdx = (1:length(OutImg))';

V{1}= V0;
V{2}= V1;
for stage = 1:PCANet.NumStages
    [OutImg,ImgIdx] = PCA_output(OutImg, ImgIdx, PCANet.PatchSize(stage), PCANet.NumFilters(stage), V{stage}); 
    path = strcat('./特征图/10class/',num2str(i),'_',num2str(stage),'.mat');
    save(path,'OutImg','ImgIdx');
end

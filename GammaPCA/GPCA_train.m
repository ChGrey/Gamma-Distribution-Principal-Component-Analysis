function [V,OutImg_i] = PCANet_train(InImg,GPCA,IdtExt)
% =======INPUT=============
% InImg     Input images (cell); each cell can be either a matrix (Gray) or a 3D tensor (RGB)  
% PCANet    PCANet parameters (struct)
%       .NumStages      
%           the number of gpca layers; e.g., 1  
%       .PatchSize
%           the patch size (filter size) for square patches; e.g., 17,
%           means the final size of GPCA kernel
%       .NumFilters
%           the number of filters in each stage; e.g., 2, 
%           means 2 principal components are considered
% IdtExt    a number in {0,1}; 1 do feature extraction, and 0 otherwise  
% =======OUTPUT============
% f         PCANet features (each column corresponds to feature of each image)
% V         learned PCA filter banks (cell)
% BlkIdx    index of local block from which the histogram is compuated
% =========================

addpath('./Utils')

if length(GPCA.NumFilters)~= GPCA.NumStages;
    display('Length(GPCA.NumFilters)~=GPCA.NumStages')
    return
end
NumImg = length(InImg);     %10000

V = cell(GPCA.NumStages,1);

OutImg = InImg; 
ImgIdx = (1:NumImg)';
% clear InImg; 

for stage = 1:GPCA.NumStages
    display(['Computing GPCA filter bank and its outputs at stage ' num2str(stage) '...'])
    
    V{stage} = GPCA_FilterBank(OutImg, GPCA.PatchSize(stage), GPCA.NumFilters(stage)); % compute GPCA filter banks
    
    if stage ~= GPCA.NumStages % compute the GPCA outputs only when it is NOT the last stage
        [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
            GPCA.PatchSize(stage), GPCA.NumFilters(stage), V{stage});  
    end
end

if IdtExt == 1 % enable feature extraction
    
    f = cell(NumImg,1); % compute the GPCA training feature one by one 
    
    for idx = 1:NumImg
        OutImgIndex = ImgIdx==idx; % select feature maps corresponding to image "idx" (outputs of the-last-but-one GPCA filter bank) 
        
        [OutImg_i,~] = GPCA_output(OutImg(OutImgIndex), ones(sum(OutImgIndex),1),GPCA.PatchSize(end), GPCA.NumFilters(end), V{end});  % compute the last PCA outputs of image "idx"
        
        % save original image
%         path = strcat('./feature_map/',cls_name,'/');
%         if ~exist(path)
%             mkdir(path);
%         end
%         origin_image = OutImg{idx};
%         name = strcat(path,num2str(idx),'_0.png');
%         imwrite(origin_image,char(name));
%             
%         % save feature map
%         for i=1:size(OutImg_i,1)
%             feature = mat2gray(OutImg_i{i});
%             name = strcat(path,num2str(idx),'_',num2str(i),'.png');
%             imwrite(feature,char(name));
%         end 
%         
%         % save .mat
%         path = strcat('./output/',cls_name,'/');
%         if ~exist(path)
%             mkdir(path);
%         end
%         mixed_output = zeros(128, 128, 3);
%         mixed_output(:,:,1) = mat2gray(OutImg{idx});
%         mixed_output(:,:,2) = mat2gray(OutImg_i{1});
%         mixed_output(:,:,3) = mat2gray(OutImg_i{2});
%         mixed_output = mixed_output;
%         save_path = char(strcat(path,num2str(idx)));
%         save(save_path,'mixed_output');
        
        OutImg(OutImgIndex) = cell(sum(OutImgIndex),1);
       
    end 
end







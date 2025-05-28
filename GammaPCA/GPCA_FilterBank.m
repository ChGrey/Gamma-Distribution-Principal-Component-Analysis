function V = GPCA_FilterBank(InImg, PatchSize, NumFilters) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of GPCA filters in the bank.
% =======OUTPUT============
% V                GPCA filter banks, arranged in column-by-column manner
% =========================

addpath('./Utils')

%% Training set to learn GPCA filter banks
ImgZ = length(InImg); % number of images 
mag = (PatchSize-1)/2;
MaxSamples = 5000;
NumRSamples = min(ImgZ, MaxSamples); 
RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);

%% Learning GPCA filters (V)
NumChls = size(InImg{1},3); % channel of images
Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);   

for i = RandIdx % 1:ImgZ
    [ImgX, ImgY, NumChls] = size(InImg{i});
    img = zeros(ImgX+PatchSize-1,ImgY+PatchSize-1, NumChls);
    img((mag+1):end-mag,(mag+1):end-mag,:) = InImg{i}; 
    
    im = im2col_mean_removal(img,PatchSize); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    
    Rx = Rx + im*im';
end

%% Conventional PCA
%tic;
%Rx = Rx/(NumRSamples*size(im,2));
%[E D] = eig(Rx);
%[trash ind] = sort(diag(D),'descend');
%V = E(:,ind(1:NumFilters));  % principal eigenvectors 

%% GPCA
tic;
[~,U,~,~]=Gamma_PCA_MM_upgrade(Rx',NumFilters);% here!!!!
SVD_time = toc;
fprintf('\nSVD time: %.2f secs.\n', SVD_time);
V = U;


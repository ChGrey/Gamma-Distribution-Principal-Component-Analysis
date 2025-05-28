clc;clear;
addpath('./Utils')

kernel = load('.\kernels\0_90\ks17.mat');
V = kernel.V0;

picture = imread('.\imgs\2S1.jpg');
image = double(picture(:,:,1))/255;

PatchSize = sqrt(size(V,1));
NumFilters = size(V,2);

mag = floor((PatchSize-1)/2);     
OutImg = cell(NumFilters,1); 
cnt = 0;
[ImgX, ImgY] = size(image);
img = zeros(ImgX+PatchSize-1,ImgY+PatchSize-1);
img((mag+1):end-mag,(mag+1):end-mag,:) = image;  
        
im = im2col_mean_removal(img,PatchSize); % collect all the patches of the ith image in a matrix, and perform patch mean removal
for j = 1:NumFilters
    cnt = cnt + 1;
    OutImg{cnt} = mat2gray(reshape(V(:,j)'*im,ImgX,ImgY));  % convolution output
end
subplot(3,2,1),imshow(mat2gray(picture(:,:,1)));
subplot(3,2,2),imhist(mat2gray(picture(:,:,1))),ylabel('Frequency','FontSize',14,'FontName','Times New Roman'),ylim([0 600]);     
subplot(3,2,3),imshow(OutImg{1});
subplot(3,2,4),imhist(OutImg{1}),ylabel('Frequency','FontSize',14,'FontName','Times New Roman'),ylim([0 600]);     
subplot(3,2,5),imshow(OutImg{2});
subplot(3,2,6),imhist(OutImg{2}),ylabel('Frequency','FontSize',14,'FontName','Times New Roman'),ylim([0 600]);        
clc;clear;
load('.\����ͼ\10class\10_1.mat')
index = [1,5,20];
for i = 1:length(index)
    X = OutImg(ImgIdx == index(i));
    for j = 1:length(X)
        path = strcat('.\����ͼ\10class\',num2str(i),'_',num2str(j),'.jpg');
        imwrite(mat2gray(X{j}),path);
    end
end

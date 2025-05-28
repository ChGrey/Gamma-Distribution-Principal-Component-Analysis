clc;clear;
load('.\ÌØÕ÷Í¼\10class\10_1.mat')
index = [1,5,20];
for i = 1:length(index)
    X = OutImg(ImgIdx == index(i));
    for j = 1:length(X)
        path = strcat('.\ÌØÕ÷Í¼\10class\',num2str(i),'_',num2str(j),'.jpg');
        imwrite(mat2gray(X{j}),path);
    end
end

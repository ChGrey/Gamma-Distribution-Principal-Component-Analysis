function im = im2col_mean_removal(imag,PatchSize)

[N,M]=size(imag);
mag = (PatchSize-1)/2;

im2col=ones(PatchSize*PatchSize,(N-PatchSize+1)*(M-PatchSize+1)/4);
col=1;
for j=1+mag:N-mag
    for i=1+mag:M-mag
        temp=imag(i-mag:i+mag,j-mag:j+mag);
        im2col(:,col)=reshape(temp,PatchSize*PatchSize,1);
        col=col+1;
    end
end
im=im2col;
    
    
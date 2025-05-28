function [XP,U,num,DM]=Gamma_PCA_MM_upgrade(X,k)

[n,d]=size(X);
mm=99999999999999999999999999999999;%tuning parameter, which should be a very large positive number

% Allocate memory space
HTheta=zeros(n,d);
mu0=zeros(d,1);
U0=zeros(d,k);

index=find(X==0);
X(index)=mm;

%%%%%%%%%%%%%%%============ Initialization of U and mu ================%%%%%%%%%%%%%%%%
HTheta=-(1./X);
[Hr,Hc]=size(HTheta);
if Hr==1
    mu0=HTheta;
else
    mu0=mean(HTheta);% Calculate the mean by column 1*d
end
mu0=mu0';%% d*1
[~,~,V]=svds(HTheta,k);% SVD 
U0=V;%% The first k right singular vectors d*k
U_max=U0;
miu_max=mu0;

M=300e34;%% Tuning parameter for iteration

U1=U0;
mu1=mu0;
num=0;num2=-1;

while num>-1
   Tm=zeros(d,1);
   um=zeros(d,1);
   tTheta=zeros(n,d);
   
    %%%%%%%%%%%%%%%%%%%%%%%%============  First step:calculating the intermediate parameters ================%%%%%%%%%%%%%%%%%%%%%%%%%%
    UUT = U1*U1';
    tTheta = (-1./(mu1 + UUT*(X'-mu1)))';  
    
    vt=zeros(1,n);
    qt=zeros(1,n);
    zt=zeros(n,d);
    pt=zeros(n,d);
    
    b_2 = 1./(tTheta.^2);
    b_4 = 1./(tTheta.^4);
    vt=max(b_2');
    qt=vt.*max(b_4');  
    zt=tTheta+(1./repmat(vt.',1,d)).*(1./tTheta+X);
    pt=zt./(tTheta.^2);
    1;
    clear tTheta exp_tTheta;
    %%%%%%%%%%%%%%%%%%%%%%%%%%============  Second step: calculating mu(t+1) ================%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mu2=zeros(d,1);  
    I=ones(n,1);

    IQI = sum(qt); %% IQI is equivalent to the sum of all elements in matrix Qt
    mu2=1/IQI*(-X*UUT-pt)'*qt'; %%d*1
    1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%============  Third step: calculating the deviation M ================%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    M12=zeros(n,d);
    M12=(-X+I*(mu2)')*(U1*U1').*sqrt(qt)' - (pt+I*(mu2)').*sqrt(qt)';
    M=norm((M12),'fro');
    clear M12;
    num=num+1;
    DM(num)=M;
    fprintf(['\n num = ', num2str(num) ,'...M = ' ,num2str(M),'\n']);
    if isnan(M)||M>1e+140
        num2=find(DM==min(DM));
        if num2==1
            U1=U0;mu1=mu0;break;
        else
            U1=U0;mu1=mu0;num=0;DM=[];
        end
        continue;
    end
    
   %
    if num>2
     if num==num2 
         break;
     elseif (abs(DM(num)-DM(num-1))/abs(DM(num-1))<0.001)&&(abs(DM(num-1)-DM(num-2))/abs(DM(num-2))<0.001) %%%%%% 前后三次结果M变化不大时跳出循环，精度0.01
         break;
         1;
     end
    end
    %
    
    if  M<=min(DM)
        U_max=U1;
        miu_max=mu2;
    end
    if num>49
        U1=U_max;
        mu1=miu_max;
        break;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%============  Forth step: calculating U(t+1) ================%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    HThetac=zeros(n,d);
    Ztc=zeros(n,d);
    EU=zeros(d,d);
    U2=zeros(d,k);
    
    HThetac=-X+I*(mu2)';%% n*d
    Ztc=pt+I*(mu2)';%% n*d
    clear zt;
    EU= (HThetac)'*(Ztc.*qt') + (Ztc)'*(HThetac.*qt') - (HThetac)'*(HThetac.*qt');

    EU=(EU+EU.')/2;
    clear HThetac Ztc; 

    [~,~,U2]=svds(EU,k);

    1;  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%============  Fifth setp: t=t+1 ================%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mu1=mu2;
    U1=U2;
    1;  
end
fprintf('\n saving kernels... \n')
U1;%%d*k
mu1;%%d*1
DM;

U = U1;
mu = mu1;
XP=(X-I*mu1')*U1;
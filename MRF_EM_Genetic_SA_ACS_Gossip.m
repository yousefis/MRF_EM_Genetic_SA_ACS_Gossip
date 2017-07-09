
%############################################################################################
%#																							#
%#																							#
%#      This code is developed by: Sahar Yousefi, syousefi(-at-)ce.sharif.edu		        #
%#																							#
%#																							#
%############################################################################################
%	If you use this code, please cite one (or more) of these papers:				
%	Yousefi, Sahar, Reza Azmi, and Morteza Zahedi. "Brain tissue segmentation in MR images based on a hybrid of MRF and social algorithms." Medical image analysis 16.4 (2012): 840-848.
%	Yousefi, Sahar, Morteza Zahedi, and Reza Azmi. "3D MRI brain segmentation based on MRF and hybrid of SA and IGA." Biomedical Engineering (ICBME), 2010 17th Iranian Conference of. IEEE, 2010.
%	Ahmadvand, Ali, Sahar Yousefi, and M. T. Manzuri Shalmani. "A novel Markov random field model based on region adjacency graph for T1 magnetic resonance imaging brain segmentation." International Journal of Imaging Systems and Technology 27.1 (2017): 78-88.


function MRF_EM_Genetic_SA_ACS_Gossip(imanm,man_seg,k)
%
%   In this project the below algorithms for brain MRI segmentation are implemented:
%   	1- Expectation Maximization
%		2- Traditional MRF models contain:
%			MMD_Metropolis
%			ICM
%			Gibbs
%		3- MetropolisGenetic
%		4- MetropolisACS
%
%
%
%   Input:
%          imanm: gray color image name
%          k: Number of classes 
%   Output:
%          mask: clasification image mask
%          mu: vector of class means 
%          v: vector of class variances
%          p: vector of class proportions   

clc
close all 
fclose('all')
%******************************
global t beta alpha T0 C  noRegion P V MU
global height width outcomeFile
noRegion=k;

T0=4;
C=.98;
t=.1;
alpha=.1;

path='F:\\Thesis\\Dataset\\harvard dataset\\convertHarvard\\4-20 Normal Subjects, T1-Weighted Scans with Segmentations\\img\\5_8\';
segpath='F:\\Thesis\\Dataset\\harvard dataset\\convertHarvard\\4-20 Normal Subjects, T1-Weighted Scans with Segmentations\\seg\5_8\\';
%******************************
outcomeFile=strcat(path,num2str(imanm),'nD24.txt');
fid = fopen(outcomeFile,'w');
fclose(fid);
%******************************

 
imgExt='.bmp';
ima=imread([path num2str(imanm) imgExt]);
if isrgb(ima)
    ima=rgb2gray(ima);
end
% ima = imnoise(ima,'gaussian',0,0.005);
% ima=modifyImg();
% ima=imnoise(ima,'gaussian',0,0.2);
[ima,rect] = imcrop(ima);
% ima=(imadjust(ima,[0.2 0.7],[]));
close 
overSliceImg=imcrop(imread([path num2str(str2num(imanm)-1) imgExt]),rect);
if isrgb(overSliceImg)
    overSliceImg=rgb2gray(overSliceImg);
end
overSliceImgSeg=imcrop(imread([segpath num2str(str2num(man_seg)-1) imgExt]),rect);
if isrgb(overSliceImgSeg)
    overSliceImgSeg=rgb2gray(overSliceImgSeg);
end

man_seg_ima=imcrop(imread([segpath num2str(man_seg) imgExt]),rect);
if isrgb(man_seg_ima)
    man_seg_ima=rgb2gray(man_seg_ima);
end

undrSliceImg=imcrop(imread([path num2str(str2num(imanm)+1) imgExt]),rect);
if isrgb(undrSliceImg)
    undrSliceImg=rgb2gray(undrSliceImg);
end 

undrSliceImgSeg=imcrop(imread([segpath num2str(str2num(man_seg)+1) imgExt]),rect);
if isrgb(undrSliceImgSeg)
    undrSliceImgSeg=rgb2gray(undrSliceImgSeg);
end

[height,width]=size(ima);
hierarchy=[reshape(undrSliceImg,1,height*width);reshape(ima,1,height*width);reshape(overSliceImg,1,height*width)];
% hierarchy=[reshape(undrSliceImgSeg,1,height*width);reshape(ima,1,height*width);reshape(overSliceImgSeg,1,height*width)];
%****************************** 
P=[ ];
V=[ ];
MU=[ ];
%******************************
for slc=1:3
    grayimg=reshape( hierarchy(slc,:),height,width);
    [emres,p,mu,v]=EM(grayimg,noRegion,height,width);
    P=[P;p];
    V=[V;v];
    MU=[MU;mu];
    classIdHierarchy(slc,:)=reshape(emres,1,height*width);
end
 

method=1;

% clc
JC=[];KI=[];
[JC(1,:),KI(1,:)]=compute_jaccard(man_seg_ima,reshape(classIdHierarchy(2,:),height,width));
%******************************
metroACSRes=MetropolisACS(noRegion,hierarchy,classIdHierarchy,V,MU); 
[JC(2,:),KI(2,:)]=compute_jaccard(man_seg_ima,metroACSRes);
%******************************
% metroGeneticRes=MetropolisGenetic(noRegion,hierarchy,classIdHierarchy,V,MU); 
% [JC(3,:),KI(3,:)]=compute_jaccard(man_seg_ima,metroGeneticRes);
%  
%****************************** 
 
[mmdRes,metropolisRes]=MMD_Metropolis(noRegion,hierarchy,classIdHierarchy,V,MU);
[JC(4,:),KI(4,:)]=compute_jaccard(man_seg_ima,metropolisRes);

% figure,plot(JC)
% xlabel('JC')
% figure,plot(KI)
% xlabel('KI')

%==========================================================================
function [emres,p,mu,v]=EM(ima,noRegion,height,width)
global  outcomeFile
k=noRegion; 
% check image
ima=double(ima); 
copy=ima;           % make a copy
ima=ima(:);         % vectorize ima
mi=min(ima);        % deal with negative 
ima=ima-mi+1;       % and zero values
m=max(ima);
% create image histogram
h=histogram(ima);
x=find(h);
h=h(x);
x=x(:);h=h(:);
% initiate parameters
mu=(1:k)*m/(k+1);
v=ones(1,k)*m;
p=ones(1,k)*1/k;
% start process
sml = mean(diff(x))/1000;
% Create movie file with required parameters
outFileName = 'simplexmovie1';
fps= 5;
outfile = sprintf('%s',outFileName)
mov = avifile(outfile,'fps',fps,'quality',100,'compression','none');

t1=clock;
fig1=figure(1);
set(fig1,'Color',[1 1 1])
while(1) 
    % Expectation
    prb = distribution(mu,v,p,x);  %p(x,y|theta)=p(yi)
    scal = sum(prb,2)+eps;        % 
    loglik=sum(h.*log(scal));      %Q(theta,theta(i-1))
    %------
    clf
    bar(x,h);
    hold on
    plot(x,sum(prb,2),'r-.')
    hold on 
%     bar(x,prb,'g')
    plot(x,prb,'g--')
    legend('Intensity Distribution','Sum of Gassian Mixture Model','Gassian Mixture Model');
    drawnow 
    F = getframe(fig1);
    mov = addframe(mov,F);
%     pause(.01)
    %------
    
    %Maximizarion 
    for j=1:k
        pp=h.*prb(:,j)./scal;
        p(j) = sum(pp);
        mu(j) = sum(x.*pp)/p(j);
        vr = (x-mu(j));
        v(j)=sum(vr.*vr.*pp)/p(j)+sml;
    end
    p = p + 1e-3;
    p = p/sum(p);
    %------------------------------------------------------------------
    % Exit condition
    prb = distribution(mu,v,p,x);
    scal = sum(prb,2)+eps;
    nloglik=sum(h.*log(scal)); 
%     (nloglik-loglik)
    if(abs(nloglik-loglik)<0.0001) 
        break;
    end; 
    clf 
    bar(x,h);
    hold on
    plot(x,sum(prb,2),'r-.')
    hold on 
%     bar(x,prb,'g')
    plot(x,prb,'g--')
    legend('Intensity Distribution','Sum of Gassian Mixture Model','Gassian Mixture Model');
    drawnow
    F = getframe(fig1);
    mov = addframe(mov,F);
%     pause(.01)
end 
t2=clock;
emtime=elapsedTime(t2,t1); 
% save movie 
mov = close(mov); 

fid = fopen(outcomeFile, 'a+');
fwrite(fid, strcat('Time elapsed EM: ',num2str(emtime),','), 'char'); 
fclose(fid);

mask=calculateMask(mu,mi,copy,k,v,p);
figure,
% subplot(1,2,1),
imshow(mat2gray(reshape(ima,height,width)),'InitialMagnification',300);
title('Original image'); 
% subplot(1,2,2), 
imshow(mat2gray(mask),'InitialMagnification',300);
title(strcat('EM segmented image (Number of classes=',num2str(k),')'));
emres=mask; 
drawnow 
% result: EM segmentation
%==========================================================================
function mask=calculateMask(mu,mi,copy,k,v,p) 
% calculate mask
mu=mu+mi-1;   % recover real range
s=size(copy);
mask=zeros(s);
for i=1:s(1)
    for j=1:s(2)
        for n=1:k
            c(n)=distribution(mu(n),v(n),p(n),copy(i,j)); 
        end
        a=find(c==max(c)); 
        mask(i,j)=a(1); 
    end
end
%==========================================================================
function y=distribution(m,v,g,x)
x=x(:);
m=m(:);
v=v(:);
g=g(:);
for i=1:size(m,1)
   d = x-m(i);
   amp = g(i)/sqrt(2*pi*v(i));
   y(:,i) = amp*exp(-0.5 * (d.*d)/v(i));
end
%==========================================================================
function[h]=histogram(datos)
datos=datos(:);
ind=find(isnan(datos)==1);
datos(ind)=0;
ind=find(isinf(datos)==1);
datos(ind)=0;
tam=length(datos); 
m=ceil(max(datos))+1;
h=zeros(1,m);
for i=1:tam,
    f=floor(datos(i));    
    if(f>0 & f<(m-1))        
        a2=datos(i)-f;
        a1=1-a2;
        h(f)  =h(f)  + a1;      
        h(f+1)=h(f+1)+ a2;                          
    end;
end;
h=conv(h,[1,2,3,2,1]);
h=h(3:(length(h)-2));
h=h/sum(h);
%==========================================================================
function singleton=Singleton(rr,cc,label,grayimg,V,MU)
mu=MU(2,:);
v=V(2,:);
singleton=log(sqrt(2*pi*v(label))+eps)+power(double(grayimg(rr,cc))-double(mu(label)),2)./(2*v(label)+eps);
% singleton=0;
%==========================================================================
function doubelton=Doubleton(rr,cc,label,classID)
global  height width  
beta=1;
doubelton=0;
if (~isequal(rr,height))
    if (label==classID(rr+1,cc))
        doubelton=doubelton-beta;
    else doubelton=doubelton+beta;
    end
    %---
    if (~isequal(cc,1))
        if (label==classID(rr+1,cc-1))
            doubelton=doubelton-beta;
        else doubelton=doubelton+beta;
        end
    end
    %---
    if (~isequal(cc,width))
        if (label==classID(rr+1,cc+1))
            doubelton=doubelton-beta;
        else doubelton=doubelton+beta;
        end
    end
end
if (~isequal(cc,width))
    if (label==classID(rr,cc+1))
        doubelton=doubelton-beta;
    else doubelton=doubelton+beta;
    end 
end
if (~isequal(rr,1))
    if (label==classID(rr-1,cc))
        doubelton=doubelton-beta;
    else doubelton=doubelton+beta;
    end
    %---
    if (~isequal(cc,1))
        if (label==classID(rr-1,cc-1))
            doubelton=doubelton-beta;
        else doubelton=doubelton+beta;
        end
    end
    %---
    if (~isequal(cc,width))
        if (label==classID(rr-1,cc+1))
            doubelton=doubelton-beta;
        else doubelton=doubelton+beta;
        end
    end
end
if (~isequal(cc,1))
    if (label==classID(rr,cc-1))
        doubelton=doubelton-beta;
    else doubelton=doubelton+beta;
    end
end
doubelton=double(doubelton);
%==========================================================================
function thirdletons=Thirdleton(rr,cc,label,classIdHierarchy,height,width)
gamma=.5;
thirdletons=0.0;
overCid=reshape(classIdHierarchy(1,:),height,width);
underCid=reshape(classIdHierarchy(3,:),height,width);
if label==overCid(rr,cc)
    thirdletons=thirdletons-gamma;
else thirdletons=thirdletons+gamma;
end

if label==underCid(rr,cc)
    thirdletons=thirdletons-gamma;
else thirdletons=thirdletons+gamma;
end

%==========================================================================
function selectiveEnergy=CalculateSelectiveEnergy(height,width,hierarchy,classIdHierarchy,V,MU,three_d,minjcross,maxjcross)
grayimg=reshape(hierarchy(2,:),height,width);
X=reshape(classIdHierarchy(2,:),height,width);
singletons = 0.0;
doubletons = 0.0;
thirdletons=0.0;

for i=minjcross:maxjcross
    rr=floor((i-1)./width)+1;
    cc=i-(rr-1).*width;
    label=X(rr,cc);
    singletons=singletons+Singleton(rr,cc,label,grayimg,V,MU);
    doubletons=doubletons+Doubleton(rr,cc,label,X);
    if three_d
        thirdletons=thirdletons+Thirdleton(rr,cc,label,classIdHierarchy,height,width);
    end
end
selectiveEnergy=singletons + doubletons/2+thirdletons;%/2
%==========================================================================
function totalEnergy=CalculateEnergy(height,width,hierarchy,classIdHierarchy,V,MU,three_d)
grayimg=reshape(hierarchy(2,:),height,width);
X=reshape(classIdHierarchy(2,:),height,width);
singletons = 0.0;
doubletons = 0.0;
thirdletons=0.0;
for rr=1:height
    for cc=1:width
        label=X(rr,cc);
        singletons=singletons+Singleton(rr,cc,label,grayimg,V,MU);
        doubletons=doubletons+Doubleton(rr,cc,label,X);
        if three_d
            thirdletons=thirdletons+Thirdleton(rr,cc,label,classIdHierarchy,height,width);
        end
    end 
end
totalEnergy=singletons + doubletons/2+thirdletons/2;
%==========================================================================
function local_energy=LocalEnergy(rr,cc,label,grayimg,classIdHierarchy,height,width,V,MU,three_d)
local_energy=Singleton(rr,cc,label,grayimg,V,MU) +... 
Doubleton(rr,cc,label,reshape(classIdHierarchy(2,:),height,width));
if three_d
    local_energy=local_energy+Thirdleton(rr,cc,label,classIdHierarchy,height,width);
end
%==========================================================================
function classID=InitOutImage(noRegion,grayimg,V,MU)
global height width   v mu
figure,imshow(grayimg,'InitialMagnification',500);
mu=[];
v=[];
for k=1:noRegion
    I=imcrop; 
    mu(k)=mean(mean(I));
    v(k)=var(var(double(I)));
end
figure,
classID=ones(height,width);
for rr=1:height
    for cc=1:width 
        e=Singleton(rr,cc,1,grayimg,V,MU);
        for nr=2:noRegion
            e2=Singleton(rr,cc,nr,grayimg,V,MU);
            if(e2<e)
                e=e2;
                classID(rr,cc)=nr;
            end
        end
    end
    imshow(mat2gray(classID),'InitialMagnification',500)
    drawnow
end
%==========================================================================
function [mmd_classID,metro_classID]=MMD_Metropolis(noRegion,hierarchy,classIdHierarchy,V,MU)
global t alpha C 
global height width outcomeFile
grayimg=reshape(hierarchy(2,:),height,width);
three_d=0;
rand('state', 100*sum(clock));
mmd_kszi=log(alpha);%This is for MMD. When executing
K = 0;
T = 4;
itr=0;
mmd_vect_E_old=[];
mmd_vect_summa_deltaE=[];

metro_vect_summa_deltaE=[];

mmd_classID=reshape(classIdHierarchy(2,:),height,width);
metro_classID=mmd_classID;
% t1=clock;
tic
mmd_E_old = CalculateEnergy(height,width,hierarchy,classIdHierarchy,V,MU,three_d);
metro_vect_E_old=[mmd_E_old];
metro_E_old = mmd_E_old;

mmd_flag=0;
metro_flag=0;

while(1)
    itr=itr+1;
    mmd_summa_deltaE = 0;
    metro_summa_deltaE = 0;
    for rr=1:height 
        for cc=1:width
%             if (noRegion==2)
%                 mmd_r=1-mmd_classID(rr,cc);
%             else
%                 mmd_r=floor(1+(noRegion)*rand);
%             end
            
            if (noRegion==2)
                metro_r=1-metro_classID(rr,cc);
            else
                metro_r=floor(1+(noRegion)*rand);
            end
           
%             if((~mmd_flag)&&(mmd_kszi<=(LocalEnergy(rr,cc, mmd_classID(rr,cc),grayimg,classIdHierarchy,height,width,V,MU,three_d)-LocalEnergy(rr,cc,mmd_r,grayimg,classIdHierarchy,height,width,V,MU,three_d)) / T))%
%                 mmd_summa_deltaE=mmd_summa_deltaE+abs(LocalEnergy(rr,cc,mmd_r,grayimg,classIdHierarchy,height,width,V,MU,three_d) - LocalEnergy(rr,cc,mmd_classID(rr,cc),grayimg,classIdHierarchy,height,width,V,MU,three_d));
%                 mmd_E_old=mmd_E_old-LocalEnergy(rr,cc,mmd_classID(rr,cc),grayimg,classIdHierarchy,height,width,V,MU,three_d)+LocalEnergy(rr,cc,mmd_r,grayimg,classIdHierarchy,height,width,V,MU,three_d);
%                 mmd_classID(rr,cc) = mmd_r;
%             end
            
            metro_kszi=log(rand);
            deltaE=(LocalEnergy(rr,cc, metro_classID(rr,cc),grayimg,classIdHierarchy,height,width,V,MU,three_d)-LocalEnergy(rr,cc,metro_r,grayimg,classIdHierarchy,height,width,V,MU,three_d));
            if((~metro_flag)&&(metro_kszi<= deltaE/ T))%
                metro_summa_deltaE=metro_summa_deltaE+abs(deltaE);
                metro_E_old=metro_E_old-deltaE;
                metro_classID(rr,cc) = metro_r;
            end
        end
    end

%     mmd_vect_E_old=[mmd_vect_E_old,mmd_E_old];
%     mmd_vect_summa_deltaE=[mmd_vect_summa_deltaE,mmd_summa_deltaE];
    
    metro_vect_E_old=[metro_vect_E_old,metro_E_old];
    metro_vect_summa_deltaE=[metro_vect_summa_deltaE,metro_summa_deltaE];
    
    figure (30),plot(metro_vect_E_old)
    title ('Metropolis Energy Convergence')
    
    T=T*C;
    K=K+1;

%     if(mmd_summa_deltaE <= t) 
%         mmd_flag=1;
%         mmd_t2=clock;
%         mmd_emtime=elapsedTime(mmd_t2,t1);
%         mmd_itr=itr;
%     end
    if(metro_summa_deltaE <= t) 
        metro_flag=1;
%         metro_t2=toc;
        metro_emtime=toc%elapsedTime(metro_t2,t1);
        metro_itr=itr;
    end
    if metro_flag%&& mmd_flag 
        break;
    end
end

 
fid = fopen(outcomeFile, 'a+');
% fwrite(fid, strcat('Time elapsed MMD: ',num2str(mmd_emtime),', itr:',num2str(mmd_itr),', '), 'char');
fwrite(fid, strcat('Time elapsed Metropolis: ',num2str(metro_emtime),', itr:',num2str(metro_itr),', '), 'char');
fclose(fid);

% figure(33),
% plot(metro_vect_summa_deltaE,'b-o');
% hold on
% % plot(mmd_vect_summa_deltaE,'r-o');
% ylabel('\Delta E');
% xlabel('iteration');
% legend('Metropolis')%,'MMD');
% hold off

figure(31)
plot(metro_vect_summa_deltaE,'-.o')
title(['Metropolis \Sigma' '\Delta' 'E'])
xlabel('Iteration');
ylabel(['\Sigma' '\Delta' 'E']);


figure(32),
imshow(mat2gray(metro_classID),'InitialMagnification',300);%,
title('Metropolis Segmentation Result');

figure(33),
plot(metro_vect_E_old,'b-o');
hold on
% plot(mmd_vect_E_old,'r-o');
ylabel('Energy');
xlabel('Iteration');
legend('Metropolis')%,'MMD');
title('Metropolis Energy Convergence')
hold off


% figure(103),hold on
% plot(metro_vect_E_old,'-')
% title('Energy Convergence')
% xlabel('Iteration');
% ylabel('Energy');
% legend('MRF-ACS-Gossiping','MRF-SA-GA','Metropolis');


% figure(33),
% imshow(mat2gray(mmd_classID),'InitialMagnification',300);%,
% title('MMD result');

%==========================================================================
function icmRes=ICM(em,noRegion,grayimg,classID,V,MU)
global t  outcomeFile
global height width 
rand('state', 100*sum(clock));
if (em==0)
    classID=InitOutImage();
end

K=0;
% figure
itr=0;
vect_summa_deltaE=[];
vect_E_old=[];

t1=clock;
E_old = CalculateEnergy(classID,height,width,grayimg,V,MU,three_d);
while(1)
    itr=itr+1;
    summa_deltaE = 0;
    for rr=1:height
        for cc=1:width
            for r=1:noRegion
                if (LocalEnergy(rr,cc, classID(rr,cc),grayimg,classIdHierarchy,height,width,V,MU,three_d)>LocalEnergy(rr,cc,r,grayimg,classIdHierarchy,height,width,V,MU,three_d))               
                    classID(rr,cc)=r;
                end
            end
        end
    end
    E = CalculateEnergy(classID,height,width,grayimg,V,MU,three_d);
    summa_deltaE = summa_deltaE+abs(E_old-E)
    vect_summa_deltaE=[vect_summa_deltaE,summa_deltaE];
    vect_E_old=[vect_E_old,E];
    E_old = E;
    K=K+1;
%     imshow(mat2gray(classID),'InitialMagnification',500);
%     drawnow
    if(summa_deltaE <= t) 
        break; 
    end
end
t2=clock;
emtime=elapsedTime(t2,t1);


fid = fopen(outcomeFile, 'a+');
fwrite(fid, strcat('Time elapsed ICM: ',num2str(emtime),', itr:',num2str(itr),', '), 'char');
fclose(fid);

icmRes=classID;
title('ICM Segmentation');

figure(40),hold on
plot(vect_summa_deltaE,'y-.');
legend('Metropolis','MMD','ICM');
hold off
% ylabel('\Delta E');
% xlabel('iteration');
figure(41),hold on
plot(vect_E_old,'y-.');
legend('Metropolis','MMD','ICM');
hold off
% ylabel('Energy');
% xlabel('iteration');

%==========================================================================
function gibbsRes=Gibbs(em,noRegion,grayimg,classID,V,MU)
global T0 C  t outcomeFile
global height width 
rand('state', 100*sum(clock));
if (em==0)
    classID=InitOutImage();
end
K = 0;
T = T0;
% figure
itr=0;
vect_summa_deltaE=[];
vect_E_old=[];
three_d=0;

t1=clock;
E_old = CalculateEnergy(classID,height,width,grayimg,V,MU,three_d);
while(1)
    itr=itr+1;
    summa_deltaE = 0;
    for rr=1:height
        for cc=1:width
            sumE = 0.0;
            for s=1:noRegion
                Ek(s)=exp(-LocalEnergy(rr, cc, s,grayimg,classIdHierarchy,height,width,V,MU,three_d)/T);
                sumE=sumE+Ek(s);
            end
            r=rand;
            z=0;
            for s=1:noRegion
                z = z+Ek(s)/sumE;
                if(z>r)
                    classID(rr,cc)=s;
                    break;
                end
            end
        end
    end
    E = CalculateEnergy(classID,height,width,grayimg,V,MU,three_d);
    summa_deltaE =summa_deltaE+abs(E_old-E)
    vect_summa_deltaE=[vect_summa_deltaE,summa_deltaE];
    vect_E_old=[vect_E_old,E];
    if(summa_deltaE <= t)
        break;
    end
    E_old = E;
    T =T*C;         % decrease temperature
    K=K+1;	      % advance iteration counter
%     imshow(mat2gray(classID),'InitialMagnification',500);
%     drawnow;
end
t2=clock;
emtime=elapsedTime(t2,t1);


fid = fopen(outcomeFile, 'a+');
fwrite(fid, strcat('Time elapsed Gibbs: ',num2str(emtime),', itr:',num2str(itr),', '), 'char');
fclose(fid);

title('Gibbs Segmentation');
gibbsRes=classID;

figure(50),hold on
plot(vect_summa_deltaE,'gs-.');
legend('Metropolis','MMD','ICM','Gibbs');
hold off
% ylabel('\Delta E');
% xlabel('iteration');
figure(51),hold on
plot(vect_E_old,'gs-.');
legend('Metropolis','MMD','ICM','Gibbs');
hold off
% ylabel('Energy');
% xlabel('iteration');


%==========================================================================
function [Engfit,pop]=inigroup(popsize,lchromNo,classID)
pop=repmat(reshape(classID,1,lchromNo),popsize,1);
% rind=randint(1,randint(1,1,[1,popsize*lchromNo]),[1,popsize*lchromNo]);
% rval=randint(1,length(rind),[1,noRegion]);
% pop(rind)=rval;
Engfit=[];
fitEngy=fitnessEnergy(pop(1,:),hierarchy,classIdHierarchy,V,MU);
for i=1:popsize
    Engfit=[Engfit;fitEngy];
end 
% pop=randint(popsize,lchromNo,[1,noRegion]);

%==========================================================================
function metroGeneticRes=MetropolisGenetic(noRegion,hierarchy,classIdHierarchy,V,MU)
global t alpha C  outcomeFile
global height width wheelSelection
grayimg=reshape(hierarchy(2,:),height,width);
t=.1; 
%1. initialization
popsize=5;%population
wheelSelection=zeros(1,popsize);
%3.
rand('state', 100*sum(clock));
kszi=log(alpha);%This is for MMD. When executing
K = 0;
T = 100000;
mmd=0;
itr=0;
three_d=1;

goodp=classIdHierarchy(2,:);
pop=reshape(goodp,height,width);
e=CalculateEnergy(height,width,hierarchy,classIdHierarchy,V,MU,three_d);
fitEng=[1/e,e];
vect_E_old=[];
vect_E_old=[vect_E_old min(fitEng(:,2))]; 
cnt=0;
bestsofar=goodp;
bestsofar_e=e;
bestsofar_vec=[];
summa_delta=[];
disp('now!!');
t1=clock;
delta_bestsofar_e=1;
itr=1;
itr2=1;
summa_deltaE1=1;summa_deltaE2=1;
while((itr<1000))%&&(delta_bestsofar_e>.01)
    if (size(pop,3)<popsize)
        for indvno=1:popsize
            %generate population
            [fitEng,pop,summa_deltaE]=Mutation(fitEng,pop,pop(:,:,size(pop,3)),size(pop,3),noRegion,classIdHierarchy,height,width,V,MU,T,popsize,grayimg,three_d);
        end 
    else
        rand('state', 100*sum(clock));
        operation=rand(1);
        if operation>=0
            [fitEarr,ch1,ch2,indxs]=CrossOver(fitEng,pop,hierarchy,classIdHierarchy,V,MU,three_d); 
            cnt=cnt+2;
            T=T*C;
            if(ch1)
                [fitEng,pop,summa_deltaE1]=Mutation(fitEng,pop,ch1,indxs(1),noRegion,classIdHierarchy,height,width,V,MU,T,popsize,grayimg,three_d);
                cnt=cnt+1;
            end
            if (ch2)
                [fitEng,pop,summa_deltaE2]=Mutation(fitEng,pop,ch2,indxs(2),noRegion,classIdHierarchy,height,width,V,MU,T,popsize,grayimg,three_d);
                cnt=cnt+1;
            end
        end 
    end
    itr=itr+1;
    goodp=pop(:,:,find(fitEng(:,1)==max(fitEng(:,1)),1));
    E_new_goodp=min(fitEng(:,2)); 
    vect_E_old=[vect_E_old,E_new_goodp];
    if(E_new_goodp<bestsofar_e) 
%         T
        delta_bestsofar_e=abs(bestsofar_e-E_new_goodp);
        bestsofar=goodp;
        bestsofar_e=E_new_goodp;
        bestsofar_vec=[bestsofar_vec,bestsofar_e];
%         figure(20),imshow(mat2gray(goodp),'InitialMagnification',300);%,
%         drawnow
    end
    K=K+1;
    %========
    summa_delta=[summa_delta summa_deltaE1 summa_deltaE2];
%     if ~summa_deltaE1
%         zros(mode(itr2,5)+1)='t';
%     else
%         zros(mode(itr2,5)+1)='f';
%     end
%     itr2=itr2+1;
%     if ~summa_deltaE2
%         zros(mode(itr2,5)+1)='t';
%     else
%         zros(mode(itr2,5)+1)='f';
%     end
    itr2=itr2+1;
    if summa_deltaE1==0%(findstr(zros,'ttttt'))
        break;
    end
%     if (size(zros,2)==5)
%         zros(1)='';
%     end
    %========
%     if(summa_deltaE1==0||summa_deltaE2==0 ) 
%         break;
%     end
%     if delta_bestsofar_e<0.01
%         break;
%     end
%     if((T <t) ) 
%         break;
%     end
end
t2=clock; 
emtime=elapsedTime(t2,t1)
fid = fopen(outcomeFile, 'a+');
fwrite(fid, strcat('Time elapsed MetropolisGenetic: ',num2str(emtime),', itr:',num2str(itr),', '), 'char');
fclose(fid);


title('Metropolis Genetic');
metroGeneticRes=bestsofar;

figure(21)
plot(summa_delta,'-.og')
title(['MRF-SA-GA \Sigma' '\Delta' 'E'])
xlabel('Iteration');
ylabel(['\Sigma' '\Delta' 'E']);

figure(22),imshow(mat2gray(reshape(bestsofar,height,width)),'InitialMagnification',300);%,
title('MRF-SA-GA Segmentatio Result');

figure(23)
plot(vect_E_old,'go-');hold on 
legend('MRF-SA-GA Energy Convergence');
xlabel('Iteration')
ylabel('Energy')
hold off
size(bestsofar_vec)
% figure(103),hold on
% plot(vect_E_old,'-g')
% title('Energy Convergence')
% xlabel('Iteration');
% ylabel('Energy');
% legend('MRF-ACS-Gossiping','MRF-SA-GA');

%**************************************************************************
function f=findIdentical(indv,E,pop,fitEng,popsize)
f=0;
for i=1:popsize
    if fitEng(i,2)==E
        if pop(:,:,i)==indv
            f=1;
            break;
        end
    end
end
%**************************************************************************
function [parent1,parent2]=roulettewheelSelection(Engfit,popsize)
global wheelSelection
mateProb=Engfit(:,1)./sum(Engfit(:,1));
selectionNo= ceil(mateProb'*popsize);
if(wheelSelection>=selectionNo)
    wheelSelection=zeros(1,size(selectionNo,2));
end
SNo=selectionNo;
while(1)
    ind=find(SNo==max(SNo),1);
    if wheelSelection(ind)<SNo(ind)
        wheelSelection(ind)=wheelSelection(ind)+1;
        SNo(ind)=0;
        parent1=ind;
        break;
    else
        SNo(ind)=0;
        continue;
    end
end
tmp=wheelSelection(ind);
wheelSelection(ind)=selectionNo(ind);
if(wheelSelection>=selectionNo)
    wheelSelection=zeros(1,size(selectionNo,2));
end
SNo=selectionNo;
SNo(ind)=0;
while(1)
    ind=find(SNo==max(SNo),1);
    if wheelSelection(ind)<SNo(ind)        
        wheelSelection(ind)=wheelSelection(ind)+1;
        SNo(ind)=0;
        parent2=ind;
        break;
    else
        SNo(ind)=0;
        continue;
    end 
end
wheelSelection(ind)=tmp;
%**************************************************************************
function [fitEarr,ch1,ch2,indxs]=CrossOver(Engfit,pop,hierarchy,classIdHierarchy,V,MU,three_d)
Pc=0.9; 
height=size(pop(:,:,1),1);
width=size(pop(:,:,1),2);
popsize=size(pop,3);
ch1=0;ch2=0;
indxs=0;
fitEarr=[0,0];
if  rand(1)<Pc
    [i,j]=roulettewheelSelection(Engfit,popsize);
    if i~=j
        indxs(1)=i;
        indxs(2)=j;
        oldp1=pop(:,:,i);
        oldp2=pop(:,:,j); 
        [newp1,newp2,minjcross,maxjcross]=crossover(oldp1,oldp2);
        t1=Engfit(i,:);
        t2=Engfit(j,:);
%         t3=fitnessEnergy(reshape(newp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d);
%         t4=fitnessEnergy(reshape(newp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d);
        
        ind1=[minjcross-width-1,minjcross-1,minjcross,maxjcross-width-1,minjcross-width-2];
        ind2=[maxjcross+1,maxjcross+width+1,minjcross+width+1,maxjcross,minjcross+width+2];
        
        ind1(find(ind1<1))=1;
        ind2(find(ind2>height*width))=height*width;
        
        E_B=fitnessSelectiveEnergy(reshape(oldp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(1),ind1(2))+...
            fitnessSelectiveEnergy(reshape(oldp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(1),ind2(2));
        
        E_C=fitnessSelectiveEnergy(reshape(oldp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(3),ind2(3))+...
            fitnessSelectiveEnergy(reshape(oldp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(4),ind2(4));
        
        E_D=fitnessSelectiveEnergy(reshape(oldp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(5),ind1(5));
        
        E_J=fitnessSelectiveEnergy(reshape(newp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(1),ind1(2))+...
            fitnessSelectiveEnergy(reshape(newp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(1),ind2(2));
        
        E_I=fitnessSelectiveEnergy(reshape(newp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(3),ind2(3))+...
            fitnessSelectiveEnergy(reshape(newp1,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(4),ind2(4));
        
        E_H=fitnessSelectiveEnergy(reshape(oldp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(5),ind1(5));
        
        E_F=fitnessSelectiveEnergy(reshape(oldp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(1),ind1(2))+...
            fitnessSelectiveEnergy(reshape(oldp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(1),ind2(2));
        
        E_G=fitnessSelectiveEnergy(reshape(oldp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(3),ind2(3))+...
            fitnessSelectiveEnergy(reshape(oldp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(4),ind2(4));
        
        E_L=fitnessSelectiveEnergy(reshape(newp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(1),ind1(2))+...
            fitnessSelectiveEnergy(reshape(newp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind2(1),ind2(2));
        
        E_K=fitnessSelectiveEnergy(reshape(newp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(3),ind2(3))+...
            fitnessSelectiveEnergy(reshape(newp2,1,height*width),hierarchy,classIdHierarchy,V,MU,three_d,ind1(4),ind2(4));
              
        t3=[1/(t1(2)-E_B-E_C-E_D+E_J+E_I+E_H) (t1(2)-E_B-E_C-E_D+E_J+E_I+E_H)];
        t4=[1/(t2(2)-E_F-E_G-E_H+E_D+E_L+E_K) (t2(2)-E_F-E_G-E_H+E_D+E_L+E_K)];
        
        tmpt=[t1;t2;t3;t4];
        tmpp(:,:,1)=oldp1;
        tmpp(:,:,2)=oldp2;
        tmpp(:,:,3)=newp1; 
        tmpp(:,:,4)=newp2;
        ind=find(tmpt==max(tmpt(:,1)),1);
        ch1=tmpp(:,:,ind);
        fitEarr(1,:)=tmpt(ind,:);
        tmpt(ind)=0;
        ind=find(tmpt==max(tmpt(:,1)),1);
        ch2=tmpp(:,:,ind);
        fitEarr(2,:)=tmpt(ind,:);
    end
end

%==========================================================================
function [oldp1,oldp2,minjcross,maxjcross]=crossover(oldp1,oldp2) 
s=size(oldp1,1)*size(oldp1,2);
crossj1=floor((floor((s-1)*rand)+1))+1; 
crossj2=floor((floor((s-1)*rand)+1))+1;
minjcross=min(crossj1,crossj2);
maxjcross=max(crossj1,crossj2);
t=oldp1(minjcross:maxjcross);
oldp1(minjcross:maxjcross)=oldp2(minjcross:maxjcross);
oldp2(minjcross:maxjcross)=t;

%==========================================================================
function t=fitnessEnergy(p,hierarchy,classIdHierarchy,V,MU,three_d)
global height width 
eng=CalculateEnergy(height,width,hierarchy,[classIdHierarchy(1,:); p;classIdHierarchy(3,:)],V,MU,three_d);
t=[1/eng,eng];
%==========================================================================
function eng=fitnessSelectiveEnergy(p,hierarchy,classIdHierarchy,V,MU,three_d,minjcross,maxjcross)
global height width 
eng=CalculateSelectiveEnergy(height,width,hierarchy,[classIdHierarchy(1,:); p;classIdHierarchy(3,:)],V,MU,three_d,minjcross,maxjcross);

%**************************************************************************
function [Engfit,pop,summa_deltaE]=Mutation(Engfit,pop,indv,indx,noRegion,classIdHierarchy,height,width,V,MU,T,popsize,grayimg,three_d)
Pm=.05;
ch=indv; 
randMat=rand(floor(size(ch,1)),floor(size(ch,2)));
chngIndx=find(randMat<Pm);
if ~ isempty(chngIndx)
    summa_deltaE=0;
    
    rr=floor((chngIndx-1)/(width))+1; 
    cc=chngIndx-(rr-1)*(width);
    
    over=reshape(classIdHierarchy(1,:),height,width);
    undr=reshape(classIdHierarchy(3,:),height,width);
    sumE=0;
    for i=1:length(chngIndx)
        w=[rr(i)-1,cc(i)-1;rr(i),cc(i)-1;rr(i)+1,cc(i)-1;
            rr(i)-1,cc(i);rr(i)+1,cc(i);            
            rr(i)-1,cc(i)+1;rr(i),cc(i)+1;rr(i)+1,cc(i)+1;];
        tmp=find((w(:,1)>0) & (w(:,1)<=height)&(w(:,2)>0)&(w(:,2)<=width));
        wintm=[height*(cc(i)-2)+rr(i)-1;height*(cc(i)-2)+rr(i);height*(cc(i)-2)+rr(i)+1;
            height*(cc(i)-1)+rr(i)-1;height*(cc(i)-1)+rr(i)+1;
            height*(cc(i))+rr(i)-1;height*(cc(i))+rr(i);height*(cc(i))+rr(i)+1;];
        wintm=wintm(tmp);
        win=ch(wintm);
        for r=1:noRegion
            rcnt(r)=length(find(win==r)); 
        end 
        rcnt(over(rr(i),cc(i)))=rcnt(over(rr(i),cc(i)))+.5;
        rcnt(undr(rr(i),cc(i)))=rcnt(undr(rr(i),cc(i)))+.5;
        maxLabl=find(rcnt==max(rcnt));
        if length(maxLabl)>1
            maxLabl=maxLabl(randint(1,1,[1,length(maxLabl)]));
        end
        oldL=ch(rr(i),cc(i)); 
        if(maxLabl~=oldL)
            deltaU=LocalEnergy(rr(i),cc(i),oldL,grayimg,[classIdHierarchy(1,:);reshape(ch,1,height*width);classIdHierarchy(3,:)],height,width,V,MU,three_d)-...
                LocalEnergy(rr(i),cc(i), maxLabl,grayimg,[classIdHierarchy(1,:);reshape(ch,1,height*width);classIdHierarchy(3,:)],height,width,V,MU,three_d);
            rand('state', 100*sum(clock));
            kszi=(rand(1));
            if((deltaU>=0)||((deltaU<0)&&(kszi>exp(-deltaU/T))))
                sumE=sumE-deltaU;
                ch(rr(i),cc(i))=maxLabl;
                summa_deltaE=summa_deltaE+abs(deltaU);
            end
        end
        
    end
    newU=Engfit(indx,2)+sumE;
    if ~findIdentical(ch,newU,pop,Engfit,size(pop,3))
        if size(pop,3)<popsize            
            indx=indx+1;
        end
        pop(:,:,indx)=ch;
        Engfit(indx,:)=[1/(newU+eps),newU];
    end
end
%**************************************************************************
function k=kappaIndx(man_seg,res_seg,man_label,seg_label)
man_ind=find(man_seg==man_label) ;
seg_ind=find(res_seg==seg_label);

Dintersect=length(intersect(man_ind,seg_ind));

k=2*Dintersect/(length(man_ind)+length(seg_ind)); 
%**************************************************************************
function j=jaccrad(man_seg,res_seg,man_label,seg_label)
man_ind=find(man_seg==man_label) ;
seg_ind=find(res_seg==seg_label);

Dintersect=length(intersect(man_ind,seg_ind));
Dunion=length(union(man_ind,seg_ind));

j=Dintersect/(Dunion+eps); 

%**************************************************************************
function [JC,KI]=compute_jaccard(man_seg,res_seg)
global noRegion outcomeFile
get_man_label=[0, 129,193,255];
for seg_label=1:noRegion
    man_label=get_man_label(seg_label);
    JC(seg_label)=jaccrad(man_seg,res_seg,man_label,seg_label);
    KI(seg_label)=kappaIndx(man_seg,res_seg,man_label,seg_label);
end
fid = fopen(outcomeFile, 'a+');
fwrite(fid, strcat('jaccard: ',num2str(JC),','), 'char');
fwrite(fid, strcat('kappa: ',num2str(KI),','), 'char');
fclose(fid); 

%**************************************************************************
function et=elapsedTime(t1,t2)
h=t2(4)-t1(4); 
m=t2(5)-t1(5);
s=t2(6)-t1(6); 

et=abs(h*3600+m*60+s);

%**************************************************************************
%**************************************************************************
function metroACSRes=MetropolisACS(noRegion,hierarchy,classIdHierarchy,V,MU)
three_d=1;
alpha=.1;
roh=.1;
global height width outcomeFile

grayimg=reshape(hierarchy(2,:),height,width);
classID=reshape(classIdHierarchy(2,:),height,width);
E_old = CalculateEnergy(height,width,hierarchy,classIdHierarchy,V,MU,three_d);
E_new=E_old;
EVec=[E_old];
tua=.0001*ones(height*width,noRegion);
% eta=.0001*ones(height*width,noRegion);
for i=1:height*width
    rr=floor((i-1)/width)+1;
    cc=i-(rr-1)*width;
    for l=1:noRegion
        etaMat(i,l)=LocalEnergy(rr,cc,l,grayimg,classIdHierarchy,height,width,V,MU,three_d);
    end
end
etaMat=-etaMat+abs(min(min(-etaMat)));
eta=(etaMat);
    itr=0;
T=4;
% deltaEnrgy=1;
tic
% l=[1:noRegion];
i=[1:height*width];
rr=floor((i-1)./width)+1;
cc=i-(rr-1).*width;
 
% deltaE=1;
sparameterMat=ones(height,width);
A=[];
while (1)
    summa_deltaE=0;
    prob=tua.*eta;
    prob=prob./(repmat(sum(prob,2),1,noRegion)+eps);
    rand('state', sum(100*clock));   
    
    for i=1:height*width
        lmax = find(cumsum(prob(i,:))>=rand(1), 1);
        if lmax~=classID(rr(i),cc(i))
            deltaE=(LocalEnergy(rr(i),cc(i),classID(rr(i),cc(i)),grayimg,classIdHierarchy,height,width,V,MU,three_d)-...
                LocalEnergy(rr(i),cc(i),lmax,grayimg,classIdHierarchy,height,width,V,MU,three_d));
            rand('state', 100*sum(clock));

            if (deltaE>0)
                dis=abs(double(grayimg(rr(i),cc(i)))-double(MU(lmax)));
                distance=exp(-dis);
                tua(i,lmax)=(1-alpha)*tua(i,lmax)+alpha*distance;
                E_new=E_new-deltaE;
                classID(rr(i),cc(i))=lmax;
                summa_deltaE=summa_deltaE+abs(deltaE);
            end
        end
    end
 
%     figure(10),plot(EVec,'-r')
%     title('MRF-ACS-Gossiping Energy Convergence');
    deltaEnrgy=E_new-E_old;
    EVec=[EVec E_new];
    E_old=E_new;
    if deltaEnrgy<0 
        tua=(1-roh)*tua+roh*abs(deltaEnrgy/E_old);
    end 
    T=T*.98;
    itr=itr+1;
    %---
    if ~summa_deltaE
        zros(mode(itr,5)+1)='t';
    else
        zros(mode(itr,5)+1)='f';
    end
    if (findstr(zros,'ttttt'))
        break;
    end
    if (size(zros,2)==5)
        zros(1)='';
    end
    A=[A summa_deltaE];
    %======================================================================
    %Gossiping
    gamma=.1;
    for it=1:height*width
        rrt=floor((it-1)/width)+1;
        cct=it-(rrt-1)*width;
%         neighbors=[rrt-1 cct-1; rrt-1 cct;rrt-1 cct+1;rrt cct-1; rrt cct+1;rrt+1 cct-1; rrt+1 cct;rrt+1 cct+1];
        neighbors=[rrt-2 cct-2;rrt-2 cct-1;rrt-2 cct;rrt-2 cct+1;rrt-2 cct+2;
                   rrt-1 cct-2;rrt-1 cct-1;rrt-1 cct;rrt-1 cct+1;rrt-1 cct+2;
                   rrt cct-2;rrt cct-1;rrt cct+1;rrt cct+2;
                   rrt+1 cct-2;rrt+1 cct-1;rrt+1 cct;rrt+1 cct+1;rrt+1 cct+2;
                   rrt+2 cct-2;rrt+2 cct-1;rrt+2 cct;rrt+2 cct+1;rrt+2 cct+2;];
        ind=find(neighbors(:,1)>=1 & neighbors(:,1)<= height& neighbors(:,2)>=1 & neighbors(:,2)<=width );
        tmpIndx=[neighbors(ind,2)+(neighbors(ind,1)-1)*width];

        labels=classID(tmpIndx);
        

        lt=zeros(1,noRegion);
        for jt=1:noRegion
            lt(jt)=size(find(labels==jt),1);
        end
        lp=lt/(sum(lt)+eps);
        
        tua(it,[1:noRegion])=tua(it,[1:noRegion])+gamma.*lp;% the central site
        
        sparameter=(size(labels,1)-size(find(labels==classID(it)),1))/size(labels,1);%/tedad hamsaye ha ke too round badi entekhab mishan
        rind=randint(1,round(sparameterMat(it)*size(labels,1)),[1 size(labels,1)]);%random index
        
        deltaInt=abs(grayimg(it)-grayimg(tmpIndx(rind)));
        gammaCoef=(double(255-deltaInt)./255);
        
        tua(tmpIndx(rind,:),classID(it))=tua(tmpIndx(rind,:),classID(it))+gammaCoef.*gamma;%for neighbour
        sparameterMat(it)=sparameter;
    end
    
    %======================================================================

% (deltaEnrgy)
end 
acs_time=toc

fid = fopen(outcomeFile, 'a+');
% fwrite(fid, strcat('Time elapsed MMD: ',num2str(mmd_emtime),', itr:',num2str(mmd_itr),', '), 'char');
fwrite(fid, strcat('Time elapsed ACS: ',num2str(acs_time)));
fclose(fid);


figure(11)
plot(A,'-.or')
title(['MRF-ACS-Gossiping \Sigma' '\Delta' 'E'])
xlabel('Iteration');
ylabel(['\Sigma' '\Delta' 'E']);

metroACSRes=classID;
figure(12),imshow(mat2gray(classID),'InitialMagnification',300);
title('MRF-ACS-Gossiping Segmentation Result');

figure(13)
plot(EVec,'-or')
legend('MRF-ACS-Gossiping Energy Convergence');
hold on
xlabel('Iteration')
ylabel('Energy')

% figure(103),hold on
% plot(EVec,'-r')
% title('Energy Convergence')
% xlabel('Iteration');
% ylabel('Energy');
% legend('MRF-ACS-Gossiping');
%**************************************************************************

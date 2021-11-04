% DFNN

%% 清空
clc;
clear all;
close all;
%% 初始化Mackey-Glass函数
%训练样本
x=ones(1,4000); x(1)=1.2;
for t=18:4017
    x(t+1)=0.9*x(t)+0.2*x(t-17)/(1+x(t-17).^10);
end
x1=x(136:635); x2=x(130:629);
x3=x(124:623); x4=x(118:617);
TrainSamInN=[x1;x2;x3;x4]; 
TrainSamOutN=x(142:641);
% 测试样本
x5=x(636:1135); x6=x(630:1129);
x7=x(624:1123); x8=x(618:1117);
TestSamInN=[x5;x6;x7;x8];
TestSamOutN=x(642:1141);

%归一化
[TrainSamIn,inputps]=mapminmax(TrainSamInN,0,1);
[TrainSamOut,outputps]=mapminmax(TrainSamOutN,0,1);
TestSamIn=mapminmax('apply',TestSamInN,inputps);

%维数和样本个数计算
[InDim,TrainSamNum]=size(TrainSamIn); %InDim输入维数4，TrainSamNum训练样本数500
OutDim=size(TrainSamOut,1); % OutDim输出维数为1
TestSamNum=size(TestSamIn,2); %TestSamNum测试样本数500

%% 参数赋值
%kdmax调节kd允许的最大值；kdmin调节kd允许的最小值；gama衰减常数，调节kd；
%emax定义的最大误差；emin定义的最小误差；beta误差的收敛常数；
%width0第一条模糊规则的宽度；k重叠因子；kw 宽度调整常数；
%kerr误差下降率法修剪规则用的预设阈值
kdmax=2;            kdmin=0.25;             gama=0.98; 
emax=1.1;           emin=0.02;              beta=0.95;           
width0=1;           k=1.2;                  kw=1.1; 
% kerr=0.00025;
kerr=0.008;

All_TrainSamIn=[]; %用于存放所有训练样本的输入
All_TrainSamOut=[];%用于存放所有训练样本的输出
All_Center=[];%用于存放所有的模糊规则的中心

%% 训练DFNN
%当第一个样本到来时
All_TrainSamIn=TrainSamIn(:,1); 
All_TrainSamOut=TrainSamOut(:,1);

%根据第一个样本建立DFNN的第一条模糊规则
All_Center=TrainSamIn(:,1);%第一个样本作为中心，
Center=All_Center'; %Center用于下面的网络计算
Width(1)=width0; %宽度为预设值，对于一个输入变量，不同模糊规则共用一个宽度
RuleNum(1)=1; %RuleNum用于存储训练过程的模糊规则数，对于第一个样本，模糊规则数=1

%计算第一个样本的输出误差
a0=RBF(dist(Center,All_TrainSamIn),1./Width'); %隶属函数层输出，肯定为1，因为中心为样本点，高斯函数输出为1
%dist函数是计算Center和ALLIN之间的欧式距离，Center是1*4，All_TrainSamIn是4*1，Center的列数应与All_TrainSamIn的行数相等
%欧式距离是m维空间中两个点之间的真实距离，d=sqrt((x1-x2)^2+(y1-y2)^2)
%注意，该RBF已经将乘算子操作包含进去(dist函数)
a0=a0/sum(a0);%规范化，a0还是为1
a01=[a0 TrainSamIn(:,1)']; 
W=All_TrainSamOut/a01'; %用伪逆求输出权值，对于第一条数据，现在只有一条规则，对于TSK模型，W为1*(4+1)
NetOut=W*a01';
RMSE(1)=sqrt(sumsqr(All_TrainSamOut-NetOut)/OutDim); %sumsqr求平方和函数，第一个样本的RMSE肯定为0

%当第2个及后续样本到来时
for i=2:TrainSamNum
    i
    pause(0.01)
%     i=2;
    Current_TrainSamIn=TrainSamIn(:,i); Current_TrainSamOut=TrainSamOut(:,i);
    All_TrainSamIn=[All_TrainSamIn Current_TrainSamIn];%连续存储测试样本的输入量
    All_TrainSamOut=[All_TrainSamOut Current_TrainSamOut];%连续存储测试样本的输出量，1*N
    [r,N]=size(All_TrainSamOut); %r=1,这里r未用到，N为当前样本数据
    [s,r]=size(Center); %center永远是4列，表示输入变量是4维，r=4
    dd=dist(Center,Current_TrainSamIn); %计算当前第i个样本的输入与现有模糊规则中心之间的距离
    [d_min,ind]=min(dd); %找出最小的距离d_min
    kd=max(kdmax*gama.^(i-1),kdmin);%对kd进行动态调整
    
    %计算第i个样本的预测输出
    ai=RBF(dist(Center,Current_TrainSamIn),1./Width'); %ai是RBF层的输出
    ai=ai/sum(ai); %规范化，当只有一个规则时，规范化后为ai=1
    ai1=transf(ai,Current_TrainSamIn);%执行TSK模型，一个规则，ai1有5个值？
    NetOut=W*ai1; %网络输出
    errout=Current_TrainSamOut-NetOut; %误差=样本真实输出-网络预测输出
    e(i)=sqrt(sum(errout.*errout)/OutDim); %求取当前样本的误差,用于指导模糊规则的产生;对于单输出情况，e(i)=abs(errout)
    ke=max(emax*beta.^(i-1),emin); %动态调整ke
    
    %% 模糊规则自组织且前提参数分配
    if d_min>kd
        
        % 增加一条模糊规则
        if e(i)>ke  %如果d_min>kd并且e(i)>ke，即新进来的这第i个数据离现有中心都太远了，并且网络泛化性能不好，才需要增加一条模糊规则。
           All_Center=[All_Center Current_TrainSamIn]; %增加一条模糊规则，中心就是第i个样本的输入
           Width_new=k*d_min; %新增RBF神经元的宽度，宽度是重叠因子k*d_min
           Width=[Width Width_new];
           Center=All_Center';
           [u,v]=size(Center);
           
           %在增加模糊规则后，计算RBF单元的输出
           A=RBF(dist(Center,All_TrainSamIn),1./Width'); %RBF单元输出，包含乘操作
           A0=sum(A);%求和
           A1=A./(ones(u,1)*A0);%归一化
           A2=transf(A1,All_TrainSamIn);%A2应该是隐含层输出阵H
          
           % 误差下降率修剪
           if u*(r+1)<=N
               %计算误差下降率
               tT=All_TrainSamOut';
               PAT=A2';
               [WW,AW]=orthogonalize(PAT); %对隐含层输出阵进行正交化计算
               SSW=sum(WW.*WW)';SStT=sum(tT.*tT)';
               err=((WW'*tT)'.^2)./(SStT*SSW');
               errT=err';
               err1=zeros(u,OutDim*(r+1));
               err1(:)=errT;
               err21=err1';
               err22=sum(err21.*err21)/(OutDim*(r+1));
               err23=sqrt(err22);
               No=find(err23<kerr);
               if ~isempty(No)
                   All_Center(:,No)=[];Center(No,:)=[]; %利用误差下降率删除模糊规则
                   Width(:,No)=[];err21(:,No)=[];
                   [uu,vv]=size(Center);
                   AA=RBF(dist(Center,All_TrainSamIn),1./Width');
                   AA0=sum(AA);
                   AA1=AA./(ones(uu,1)*AA0);
                   AA2=transf(AA1,All_TrainSamIn);
                   W=All_TrainSamOut/AA2;
                   outAA2=W*AA2;
                   sse0=sumsqr(All_TrainSamOut-outAA2)/(i*OutDim);
                   RMSE(i)=sqrt(sse0);
                   RuleNum(i)=uu;
                   w2T=W';ww2=zeros(uu,OutDim*(r+1));
                   ww2(:)=w2T;
                   w21=ww2';
               else
                   
                   W=All_TrainSamOut/A2;
                   outA2=W*A2;
                   sse0=sumsqr(All_TrainSamOut-outA2)/(OutDim*i);
                   RMSE(i)=sqrt(sse0);
                   RuleNum(i)=u;
                   w2T=W';ww2=zeros(u,OutDim*(r+1));
                   ww2(:)=w2T;
                   w21=ww2';
               end
           else %这里u*(r+1)>N
               W=All_TrainSamOut/A2;
               outA2=W*A2;
               sse0=sumsqr(All_TrainSamOut-outA2)/(OutDim*i);
               RMSE(i)=sqrt(sse0);
               RuleNum(i)=u;
               w2T=W';ww2=zeros(u,OutDim*(r+1));
               ww2(:)=w2T;
               w21=ww2';
           end  % if u*(r+1)<=N
        %  
        else   %e(i)<ke 其dmin>kd,只需调整结果参数
           a=RBF(dist(Center,All_TrainSamIn),1./Width');
           a0=sum(a);a1=a./(ones(s,1)*a0);
           a2=transf(a1,All_TrainSamIn);
           W=All_TrainSamOut/a2;
           outa2=W*a2;
           sse1=sumsqr(All_TrainSamOut-outa2)/(OutDim*i);
           RMSE(i)=sqrt(sse1);
           RuleNum(i)=s;
       end  % if e(i)>ke

    else  %if d_min>kd，下面对应的就是d_min<kd，分两种情况
        
        if e(i)>ke   %e(i)>ke且d_min<kd，RBF宽度和输出权值将被同时进行更新
            Width(ind)=kw*Width(ind); %ind 是最接近当前样本的模糊规则
            aa=RBF(dist(Center,All_TrainSamIn),1./Width');
            aa0=sum(aa);aa1=aa./(ones(s,1)*aa0);
            aa2=transf(aa1,All_TrainSamIn);
            W=All_TrainSamOut/aa2;
            outaa2=W*aa2;
            sse2=sumsqr(All_TrainSamOut-outaa2)/(i*OutDim);
            RMSE(i)=sqrt(sse2);
            RuleNum(i)=s;
        else  %e(i)<ke且d_min<kd,否则只需调整结果参数,RBF宽度不需要调整
            aa1=RBF(dist(Center,All_TrainSamIn),1./Width');
            aa01=sum(aa1);aa11=aa1./(ones(s,1)*aa01);
            aa21=transf(aa11,All_TrainSamIn);
            W=All_TrainSamOut/aa21;
            outaa21=W*aa21;
            sse3=sumsqr(All_TrainSamOut-outaa21)/(OutDim*i);
            RMSE(i)=sqrt(sse3);
            RuleNum(i)=s;
        end
        
    end  %if d_min>kd
end


%% 训练集预测
TA=RBF(dist(Center,TrainSamIn),1./Width');
TA0=sum(TA); [u,v]=size(Center);
TA1=TA./(ones(u,1)*TA0);
TA2=transf(TA1,TrainSamIn);
TrainNetOut=W*TA2;
TrainNetOutN=mapminmax('reverse',TrainNetOut,outputps); %训练集输出反归一化   

%% 计算训练集RMSE、APE、精度
TrainError=TrainSamOutN-TrainNetOutN;
TrainRMSE=sqrt(sum(TrainError.^2)/TrainSamNum);
TrainAPE=sum(abs(TrainError)./abs(TrainSamOutN))/TrainSamNum;
TrainAccuracy=sum(1-abs(TrainError./TrainSamOutN))/TrainSamNum;
TrainRMSE
TrainAPE
TrainAccuracy

%% 测试集预测
A=RBF(dist(Center,TestSamIn),1./Width');
SA=sum(A); [u,v]=size(Center);
A1=A./(ones(u,1)*SA);
A2=transf(A1,TestSamIn);
TestNetOut=W*A2;
TestNetOutN=mapminmax('reverse',TestNetOut,outputps);  %测试集输出反归一化  

%% 计算测试集RMSE、APE、精度
TestError=TestSamOutN-TestNetOutN;
TestRMSE=sqrt(sum(TestError.^2)/TestSamNum);
TestAPE=sum(abs(TestError)./abs(TestSamOutN))/TestSamNum;
TestAccuracy=sum(1-abs(TestError./TestSamOutN))/TestSamNum;
TestRMSE
TestAPE
TestAccuracy

%% 绘制曲线
%训练误差曲线
figure(1)
plot(RuleNum,'k-','LineWidth',2);
title('Fuzzy rule generation');
xlabel('Input sample patterns');
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
ylim([min(RuleNum)-1 max(RuleNum)+1]);

% figure,plot(e,'k');
% title('Actual output error e(i)');
% xlabel('Input sample patterns');

figure(2)
plot(RMSE,'k-','LineWidth',2)
xlabel('训练步数','fontsize',9)
ylabel('RMSE值','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
% xlim([0 100])
%训练结果作图

figure(3)
plot(TrainSamOutN,'k-','LineWidth',2)
hold on
plot(TrainNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Fontsize',9);
xlabel('训练样本','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试结果作图
figure(4)
k=TrainSamNum+1:TrainSamNum+TestSamNum;
plot(k,TestSamOutN,'k-','LineWidth',2)
hold on
plot(k,TestNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'fontname','times new roman','Fontsize',9);
xlabel('Testing samples','fontname','times new roman','fontsize',10)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试数据误差
figure(5)
k=501:1000;
plot(k,TestError,'k-','LineWidth',2)
xlabel('Testing samples','fontname','times new roman','fontsize',10)
ylabel('prediction error','fontname','times new roman','fontsize',10)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.17 .16 .79 .80]);  %调整 XLABLE和YLABLE不会被切掉

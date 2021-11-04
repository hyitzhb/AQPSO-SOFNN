%程序：D-FNN，动态模糊神经网络，在线自组织，误差下降率，最小二乘
%Mackey-Glass测试
%周红标

%% 清空
clc;
clear all;
close all;
%% 产生Mackey-Glass数据
%训练样本
x=ones(1,4000); x(1)=1.2;
for t=18:4017
    x(t+1)=0.9*x(t)+0.2*x(t-17)/(1+x(t-17).^10);
end
x1=x(136:635); x2=x(130:629);
x3=x(124:623); x4=x(118:617);
TrainSamInN=[x1;x2;x3;x4];
TrainSamOutN=x(142:641);
[InDim,TrainSamNum]=size(TrainSamInN); %InDim输入维数4，TrainSamNum训练样本数500
OutDim=size(TrainSamOutN,1); % OutDim输出维数为1
% 测试样本
x5=x(636:1135); x6=x(630:1129);
x7=x(624:1123); x8=x(618:1117);
TestSamInN=[x5;x6;x7;x8];
TestSamOutN=x(642:1141);
TestSamNum=size(TestSamInN,2); %TestSamNum测试样本数500
%归一化
% [TrainSamIn,inputps]=mapminmax(TrainSamInN,0,1);   %TrainSamIn为训练样本输入，4*500
% [TrainSamOut,outputps]=mapminmax(TrainSamOutN,0,1);%TrainSamOut为训练样本输出,1*500
% TestSamIn=mapminmax('apply',TestSamInN,inputps);   %TestSamIn为测试样本输入,4*500

% 无需归一化
TrainSamIn=TrainSamInN;
TrainSamOut=TrainSamOutN;
TestSamIn=TestSamInN;
%% 参数赋值
%kdmax调节kd允许的最大值；kdmin调节kd允许的最小值；gama衰减常数，调节kd；
%emax定义的最大误差；emin定义的最小误差；beta误差的收敛常数；
%width0第一条模糊规则的宽度；ko重叠因子；kw宽度调整常数；
%kerr误差下降率法修剪规则用的预设阈值
kdmax=2;            kdmin=0.25;             gama=0.98;
emax=1.1;           emin=0.02;              beta=0.95;
width0=1;           ko=1.2;                 kw=0.98;
kerr=0.00015;

%% 预定义变量
TrainSamIn_All=[]; %用于存放所有训练样本的输入
TrainSamOut_All=[];%用于存放所有训练样本的输出
Center_All=[];     %用于存放所有的模糊规则的中心

tic
%% 动态产生DFNN
%对于第一个样本
TrainSamIn_All=TrainSamIn(:,1);
TrainSamOut_All=TrainSamOut(:,1);

%根据第一个样本建立DFNN的第一条模糊规则
Center_All=TrainSamIn(:,1);%第一个样本作为中心，
Center=Center_All'; %Center用于下面的网络计算
Width(1)=width0; %宽度为预设值，对于一个输入变量，不同模糊规则共用一个宽度
RuleNum_his(1)=1; %RuleNum用于记录训练过程的模糊规则数，对于第一个样本，模糊规则数=1

%计算第一个样本的输出误差
RuleUnitOut=RBF(dist(Center,TrainSamIn_All),1./Width'); %RuleUnitOut为规则层输出，肯定为1，因为中心为样本点，高斯函数输出为1
%dist函数是计算Center和TrainSamIn_All之间的欧式距离
%Center的列数应与TrainSamIn_All的行数相等，Center是1*4，All_TrainSamIn是4*1
%欧式距离是m维空间中两个点之间的真实距离，d=sqrt((x1-x2)^2+(y1-y2)^2)
%注意，该RBF已经将乘算子操作包含进去(dist函数)
NormValue=RuleUnitOut/sum(RuleUnitOut);%NormValue为规范化层输出，NormValue=1
NormValue_new=[NormValue TrainSamIn(:,1)'];  %NormValue_new为1*5???
W=TrainSamOut_All/NormValue_new'; %用伪逆求输出权值(规则的连接权值)，对于第一条数据，现在只有一条规则，对于TSK模型。
% W为输出权值，W为1*(4+1)
NetOut=W*NormValue_new'; %NetOut为网络输出，对于第1个样本，只是1个值
Error(1)=TrainSamOut(:,1)-NetOut; %误差=样本真实输出-网络预测输出
e_norm(1)=sqrt(sum(Error(1).*Error(1))/OutDim); %求取当前样本的误差,用于指导模糊规则的产生;对于单输出情况，e_norm(i)=abs(Error)
RMSE(1)=sqrt(sumsqr(TrainSamOut_All-NetOut)/(OutDim*1)); %sumsqr求平方和函数，第一个样本的RMSE肯定为0

%% 从第二个样本开始，顺序学习
for i=2:TrainSamNum
    i
    TrainSamIn_All=[TrainSamIn_All TrainSamIn(:,i)];%连续存储训练样本的输入量,4*i     %TrainSamIn(:,i) 第i个样本的输入,4*1
    TrainSamOut_All=[TrainSamOut_All TrainSamOut(:,i)];%连续存储训练样本的输出量，1*i %TrainSamOut(:,i) 第i个样本的输出,1*1
    Num_Current=size(TrainSamOut_All,2);  %Num_Current为当前样本数据个数，这里=i
    RuleNum=size(Center,1); %center永远是4列，表示输入变量是4维，RuleNum表示当前中心的数目
    dd=dist(Center,TrainSamIn(:,i)); %计算当前第i个样本的输入与现有模糊规则中心之间的距离
    [d_min,ind]=min(dd); %找出最小的距离d_min，ind是对应的中心
    kd=max(kdmax*gama.^(i-1),kdmin);%对kd进行动态调整
    
    %在当前网络结构下，计算第i个样本的预测输出
    RuleUnitOut=RBF(dist(Center,TrainSamIn(:,i)),1./Width'); %RuleUnitOut是RBF层的输出
    NormValue=RuleUnitOut/sum(RuleUnitOut); %ai是规则层输出，(已规范化)，当只有一个规则时，规范化后为ai=1
    NormValue_new=transf(NormValue,TrainSamIn(:,i));%执行TSK模型，一个规则，ai1有5个值？
    NetOut=W*NormValue_new; %网络输出
    Error(i)=TrainSamOut(:,i)-NetOut; %误差=样本真实输出-网络预测输出
    e_norm(i)=sqrt(sum(Error(i).*Error(i))/OutDim); %求取当前样本的误差,用于指导模糊规则的产生;对于单输出情况，e_norm(i)=abs(Error)
    ke=max(emax*beta.^(i-1),emin); %动态调整ke
    
    %% 模糊规则自组织，前件和后件参数调整，分四种情况
    
    %% 第一种情况,此时DFNN有较好的泛化能力且完全可以容纳观测数据，不需要做什么，或仅需更新后件参数
    %e_norm(i)<=ke表明这个系统具有较好的泛化能力
    %d_min<=kd表明这个系统中某些神经元能够聚类这个输入向量，因此FNN能够容纳这个观察数据
    if e_norm(i)<=ke && d_min<=kd
        RuleUnitOut=RBF(dist(Center,TrainSamIn_All),1./Width');            %规则层输出
        NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut)); %规范层输出
        NormValue_new=transf(NormValue,TrainSamIn_All);
        W=TrainSamOut_All/NormValue_new; %计算输出权值，8*（1+4）=40个参数
        NetOut=W*NormValue_new;
        RMSE(i)=sqrt(sumsqr(TrainSamOut_All-NetOut)/(OutDim*i)); %均方根误差，是第一个样本到目前为止系统里所有样本的RMSE
        RuleNum_his(i)=RuleNum; %当前模糊规则数
    end
    
    %% 第二种情况，所建立的DFNN具有较好的泛化能力，只有结果参数需要调整
    %e_norm(i)<=ke表明这个系统具有较好的泛化能力
    %d_min>kd表明这个系统不满足完备性，要考虑增加一条模糊规则，有问题吧？
    %这种情况下，只需要调整结果参数？
    %修改为：在此情况下，具有最小距离的那个神经元对应的隶属函数的宽度应该被放大
    if e_norm(i)<=ke && d_min>kd    
        RuleUnitOut=RBF(dist(Center,TrainSamIn_All),1./Width');
        NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut));
        NormValue_new=transf(NormValue,TrainSamIn_All);
        W=TrainSamOut_All/NormValue_new;
        NetOut=W*NormValue_new;
        RMSE(i)=sqrt(sumsqr(TrainSamOut_All-NetOut)/(OutDim*i)); %均方根误差
        RuleNum_his(i)=RuleNum; %当前模糊规则数
    end
    
    %% 第三种情况,覆盖RBF单元的泛化能力并不是很好，需要更新RBF节点的宽度和结果参数
    %e_norm(i)>ke表明这个系统泛化能力较差
    %d_min<=kd表明FNN能够容纳这个观察数据
    if e_norm(i)>ke && d_min<=kd
        Width(ind)=kw*Width(ind); %ind是最接近当前样本的模糊规则，对其宽度进行调整
        RuleUnitOut=RBF(dist(Center,TrainSamIn_All),1./Width');
        NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut));
        NormValue_new=transf(NormValue,TrainSamIn_All);
        W=TrainSamOut_All/NormValue_new;
        NetOut=W*NormValue_new;
        RMSE(i)=sqrt(sumsqr(TrainSamOut_All-NetOut)/(OutDim*i));
        RuleNum_his(i)=RuleNum; %当前模糊规则数
    end
    
    %% 第四种情况，误差大，当前样本到中心的最小距离超过了可容纳边界的有效半径，需要增加一条新的模糊规则
    %e_norm(i)>ke表明这个系统泛化能力较差
    %d_min>kd表明FNN不能够容纳这个观察数据
    if e_norm(i)>ke && d_min>kd
        Center_All=[Center_All TrainSamIn(:,i)]; %增加一条模糊规则，中心就是第i个样本的输入
        Width_new=ko*d_min; %新增RBF神经元的宽度，宽度是重叠因子ko*d_min
        Width=[Width Width_new];
        Center=Center_All';
        RuleNum=size(Center,1); %RuleNum_Current为当前RBF神经元数，即规则数
        %在增加模糊规则后，计算RBF单元的输出
        RuleUnitOut=RBF(dist(Center,TrainSamIn_All),1./Width'); %RBF单元输出，包含乘操作
        NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut));%归一化
        NormValue_new=transf(NormValue,TrainSamIn_All);%A2应该是隐含层输出阵H
        
        % 误差下降率修剪,当样本数N大于等于u*(r+1)时，才激活基于误差下降率的修剪操作，否则QR分解无法执行
        if RuleNum*(1+InDim)<=Num_Current  %RuleNum是当前模糊规则数，InDim是输入维数，Num_Current为当前样本数据
            %计算误差下降率
            tT=TrainSamOut_All';
            PAT=NormValue_new';
            [WW,AW]=orthogonalize(PAT); %对隐含层输出阵进行正交化计算
            SSW=sum(WW.*WW)';SStT=sum(tT.*tT)';
            err=((WW'*tT)'.^2)./(SStT*SSW');
            errT=err';
            err1=zeros(RuleNum,OutDim*(InDim+1));
            err1(:)=errT;
            err21=err1';
            err22=sum(err21.*err21)/(OutDim*(InDim+1));
            err23=sqrt(err22);
            No=find(err23<kerr);
            if ~isempty(No) %利用误差下降率修剪模糊规则
                Center_All(:,No)=[];Center(No,:)=[]; %修剪该模糊规则
                Width(:,No)=[];err21(:,No)=[];
                [uu,vv]=size(Center);
                AA=RBF(dist(Center,TrainSamIn_All),1./Width');
                AA0=sum(AA);
                AA1=AA./(ones(uu,1)*AA0);
                AA2=transf(AA1,TrainSamIn_All);
                W=TrainSamOut_All/AA2;
                outAA2=W*AA2;
                sse0=sumsqr(TrainSamOut_All-outAA2)/(i*OutDim);
                RMSE(i)=sqrt(sse0);
                RuleNum_his(i)=uu;
                w2T=W';ww2=zeros(uu,OutDim*(InDim+1));
                ww2(:)=w2T;
                w21=ww2';
            else  %不需要修剪的话，就执行下面
                W=TrainSamOut_All/NormValue_new;
                outA2=W*NormValue_new;
                sse0=sumsqr(TrainSamOut_All-outA2)/(OutDim*i);
                RMSE(i)=sqrt(sse0);
                RuleNum_his(i)=RuleNum;
                w2T=W';ww2=zeros(RuleNum,OutDim*(InDim+1));
                ww2(:)=w2T;
                w21=ww2';
            end
        else %这里u*(r+1)>N
            W=TrainSamOut_All/NormValue_new;
            outA2=W*NormValue_new;
            sse0=sumsqr(TrainSamOut_All-outA2)/(OutDim*i);
            RMSE(i)=sqrt(sse0);
            RuleNum_his(i)=RuleNum;
            w2T=W';ww2=zeros(RuleNum,OutDim*(InDim+1));
            ww2(:)=w2T;
            w21=ww2';
        end  % if u*(r+1)<=N
        
    end
    
   
end
TrainTime=toc

%% 训练集预测
RuleUnitOut=RBF(dist(Center,TrainSamIn),1./Width');
NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut));
NormValue_new=transf(NormValue,TrainSamIn);
TrainNetOut=W*NormValue_new;
% TrainNetOutN=mapminmax('reverse',TrainNetOut,outputps); %训练集输出反归一化   
TrainNetOutN=TrainNetOut; %无需归一化
%% 计算训练集RMSE、APE、精度
TrainError=TrainSamOutN-TrainNetOutN;
TrainRMSE=sqrt(sum(TrainError.^2)/TrainSamNum);
TrainAPE=sum(abs(TrainError)./abs(TrainSamOutN))/TrainSamNum;
TrainAccuracy=sum(1-abs(TrainError./TrainSamOutN))/TrainSamNum;
TrainRMSE
TrainAPE
TrainAccuracy

%% 测试集预测
RuleUnitOut=RBF(dist(Center,TestSamIn),1./Width');
NormValue=RuleUnitOut./(ones(RuleNum,1)*sum(RuleUnitOut));
NormValue_new=transf(NormValue,TestSamIn);
TestNetOut=W*NormValue_new;
% TestNetOutN=mapminmax('reverse',TestNetOut,outputps);  %测试集输出反归一化  
TestNetOutN=TestNetOut; %无需归一化
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
figure;
plot(RuleNum_his,'k-','LineWidth',2);
title('Fuzzy rule generation');
xlabel('Input sample patterns');
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
ylim([min(RuleNum_his)-1 max(RuleNum_his)+1]);

figure;
plot(Error,'k');
xlabel('Training samples','fontsize',9)
ylabel('Actual output error','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
ylim([-0.1 0.1])

figure;
plot(RMSE,'k-','LineWidth',2)
xlabel('训练步数','fontsize',9)
ylabel('RMSE值','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
% xlim([0 100])
%训练结果作图

figure;
plot(TrainSamOutN,'k-','LineWidth',2)
hold on
plot(TrainNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Fontsize',9);
xlabel('训练样本','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试结果作图
figure;
k=TrainSamNum+1:TrainSamNum+TestSamNum;
plot(k,TestSamOutN,'k-','LineWidth',2)
hold on
plot(k,TestNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Fontsize',9);
xlabel('测试样本','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试数据误差
figure;
k=TrainSamNum+1:TrainSamNum+TestSamNum;
plot(k,TestError,'k-','LineWidth',2)
xlabel('测试样本','fontsize',9)
ylabel('预测误差','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.17 .16 .79 .80]);  %调整 XLABLE和YLABLE不会被切掉

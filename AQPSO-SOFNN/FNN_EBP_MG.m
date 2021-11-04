%程序：RBF型FNN，伪在线，Mackey-Glass,梯度下降，样本一个一个进，外围加循环
%作者：周红标
%地点：北京工业大学科学楼
%日期：2016.4.19

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
[InDim,TrainSamNum]=size(TrainSamInN); %InDim输入维数4，TrainSamNum训练样本数500
OutDim=size(TrainSamOutN,1); % OutDim输出维数为1
% 测试样本
x5=x(636:1135); x6=x(630:1129);
x7=x(624:1123); x8=x(618:1117);
TestSamInN=[x5;x6;x7;x8];
TestSamOutN=x(642:1141);
TestSamNum=size(TestSamInN,2); %TestSamNum测试样本数500

%归一化
[TrainSamIn,inputps]=mapminmax(TrainSamInN,0,1);
[TrainSamOut,outputps]=mapminmax(TrainSamOutN,0,1);
TestSamIn=mapminmax('apply',TestSamInN,inputps);

%% 参数设置
RuleNum=10; %规则数=10
MaxEpoch=500; %最大训练次数，300步
E0=0.001; %目标误差
lr=0.01; %学习率,取0.01（1000步）；0.1（300步）

%随机产生一组中心、宽度、权值，归一化后都取rand
Center=rand(InDim,RuleNum); %隶属函数层中心
Width=ones(InDim,RuleNum); %隶属函数层宽度,取ones很重要
W=rand(RuleNum,OutDim); %规则层与输出层之间权值   

% load Center_0 
% load Width_0 
% load W_0 
% Center=Center_0;
% Width=Width_0;
% W=W_0;
%%  建模
tic
% 重复训练2000次,MaxEpoch
for epoch=1:MaxEpoch
    epoch  
%     epoch=1

%% 提取训练样本TrainSamIn，TrainSamNum
    for k=1:TrainSamNum
%       k=1 
        SamIn=TrainSamIn(:,k);          
        % 隶属函数层，模糊化
        for i=1:InDim
            for j=1:RuleNum
                MemFunUnitOut(i,j)=exp(-(SamIn(i)-Center(i,j))^2/Width(i,j)^2);
            end
        end     
        % 规则层
        RuleUnitOut=prod(MemFunUnitOut,1); %规则层输出
        % 归一化层
        RuleUnitOutSum=sum(RuleUnitOut); %规则层输出求和 
        NormValue=RuleUnitOut./RuleUnitOutSum; %归一化层输出，自组织调整NormValue
        % 输出层
        NetOut=NormValue*W; %输出层输出，即网络输出 
        Error(k)=TrainSamOut(:,k)-NetOut;%误差=期望输出-网络实际输出  e=yd-y
    
        % 梯度     
        % 权值修正量 
        AmendW=0*W;
        for j=1:RuleNum
             AmendW(j)=-Error(k)*NormValue(j);   
        end
        %中心修正量
        AmendCenter=0*Center;
        for i=1:InDim  
            for j=1:RuleNum     
                AmendCenter(i,j)=-Error(k)*W(j)*(RuleUnitOutSum-RuleUnitOut(j))*RuleUnitOut(j)*2*(SamIn(i)-Center(i,j))/(Width(i,j)^2*RuleUnitOutSum^2);
            end
        end
        % 宽度修正量
        AmendWidth=0*Width;
        for i=1:InDim 
            for j=1:RuleNum      
                AmendWidth(i,j)=-Error(k)*W(j)*(RuleUnitOutSum-RuleUnitOut(j))*RuleUnitOut(j)*2*(SamIn(i)-Center(i,j))^2/(Width(i,j)^3*RuleUnitOutSum^2);
            end
        end
       
        % 更新中心、宽度、权值
        W=W-lr*AmendW; 
        Center=Center-lr*AmendCenter;
        Width=Width-lr*AmendWidth;

    end 
   
   % 训练RMSE
   RMSE(epoch)=sqrt(sum(Error.^2)/TrainSamNum); %TrainSamNum样本个数
   
   if RMSE(epoch)<E0,break,end
   
end
TrainTime=toc
%% 训练样本预测
for k=1:TrainSamNum
    SamIn=TrainSamIn(:,k);
    % 隶属函数层，模糊化
    for i=1:InDim
        for j=1:RuleNum
            TrainMemFunUnitOut(i,j)=exp(-((SamIn(i)-Center(i,j))^2)/(Width(i,j)^2));
        end
    end
    % 规则层
    TrainRuleUnitOut=prod(TrainMemFunUnitOut); %规则层输出
    % 输出层
    TrainRuleUnitOutSum=sum(TrainRuleUnitOut); %规则层输出求和
    TrainRuleValue=TrainRuleUnitOut./TrainRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
    TrainNetOut(k)=TrainRuleValue*W; %输出层输出，即网络输出
end
TrainNetOutN=mapminmax('reverse',TrainNetOut,outputps);    
%% 测试样本预测
   for k=1:TestSamNum
       SamIn=TestSamIn(:,k);
        % 隶属函数层，模糊化
        for i=1:InDim
            for j=1:RuleNum
                TestMemFunUnitOut(i,j)=exp(-((SamIn(i)-Center(i,j))^2)/(Width(i,j)^2));
            end
        end     
        % 规则层
        TestRuleUnitOut=prod(TestMemFunUnitOut); %规则层输出          
        % 输出层
        TestRuleUnitOutSum=sum(TestRuleUnitOut); %规则层输出求和
        TestRuleValue=TestRuleUnitOut./TestRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
        TestNetOut(k)=TestRuleValue*W; %输出层输出，即网络输出
   end
TestNetOutN=mapminmax('reverse',TestNetOut,outputps);   
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
plot(RMSE,'k-','LineWidth',2)
xlabel('训练步数','fontsize',9)
ylabel('RMSE值','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
% xlim([0 100])
%训练结果作图
figure(2)
plot(TrainSamOutN,'k-','LineWidth',2)
hold on
plot(TrainNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Fontsize',9);
xlabel('训练样本','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉
%测试结果作图
figure(3)
k=501:1000;
plot(k,TestSamOutN,'k-','LineWidth',2)
hold on
plot(k,TestNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Fontsize',9);
xlabel('测试样本','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉
%测试数据误差
figure(4)
k=501:1000;
plot(k,TestError,'k-','LineWidth',2)
xlabel('测试样本','fontsize',9)
ylabel('预测误差','fontsize',9)
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.17 .16 .79 .80]);  %调整 XLABLE和YLABLE不会被切掉

%% 保存数据
TrainRMSE_GD=RMSE;
save TrainRMSE_GD TrainRMSE_GD

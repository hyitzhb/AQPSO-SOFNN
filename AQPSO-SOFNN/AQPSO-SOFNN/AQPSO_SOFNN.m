% 自适应量子粒子群优化自组织模糊神经网络
% 前件参数和结构：自适应量子粒子群
% 后件参数：最小二乘
% 混沌时间序列预测
%% 清空
clc;
clear all;
close all;
%% 数据准备
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
%测试样本
x5=x(636:1135); x6=x(630:1129);
x7=x(624:1123); x8=x(618:1117);
TestSamInN=[x5;x6;x7;x8];
TestSamOutN=x(642:1141);
TestSamNum=size(TestSamInN,2); %TestSamNum测试样本数500
%归一化
[TrainSamIn,inputps]=mapminmax(TrainSamInN,0,1);   %TrainSamIn为归一化的训练样本输入
[TrainSamOut,outputps]=mapminmax(TrainSamOutN,0,1);%TrainSamOut为归一化的训练样本输出
TestSamIn=mapminmax('apply',TestSamInN,inputps);   %TestSamIn为归一化后的测试样本输入
%% 参数设置
PopNum = 50;      %种群规模
RuleNum_max = 15; %最大模糊规则数
RuleNum_best_his = []; %记录每一代的最优模糊规则数
pop_RuleNum = round((RuleNum_max-1)*rand(PopNum,1)+1);   %每一个个体所携带的模糊规则数，PopNum行1列,1-15之间的随机整数
pop_dim = InDim*pop_RuleNum+InDim*pop_RuleNum;       %PopNum行1列,决策空间维数=中心数目+宽度数目,只对中心和宽度两种参数进行寻优
Maxstep = 200;  %最大迭代次数
pop_bound_center = [0  1];    %中心范围
pop_bound_width =  [0.4 1.2]; %宽度范围
%种群初始化
pop=zeros(2*InDim,RuleNum_max,PopNum);   %2*InDim行，RuleNum_max列，PopNum页
for i=1:PopNum
    pop(1:InDim,1:pop_RuleNum(i),i) = pop_bound_center(1)+rand(InDim,pop_RuleNum(i))*(pop_bound_center(2)-pop_bound_center(1));  %中心，前4行
    pop(InDim+1:2*InDim,1:pop_RuleNum(i),i) = pop_bound_width(1)+rand(InDim,pop_RuleNum(i))*(pop_bound_width(2)-pop_bound_width(1));  %宽度，后4行
end

%采用最小二乘获取输出权重
for i=1:PopNum
    %     i=2    %测试用
    [fit(i),Weights(i).Weights]= fitness(pop(:,:,i),pop_RuleNum(i),TrainSamIn,TrainSamOut);
    f_pbest(i) = fit(i);
end

% 计算种群适应度值
pbest = pop;   %初始化时pbest就是种群本身
gbest = zeros(2*InDim,RuleNum_max,1);
g = min(find(f_pbest == min(f_pbest(1:PopNum))));
gbest = pbest(:,:,g);     %gbest为初始化时获取的全局最优位置
f_gbest = f_pbest(g);     %f_gbest为全局最优解对应的适应度值
Weights_best= Weights(g).Weights;
RuleNum_bset=pop_RuleNum(g);
%% 进入迭代
for step = 1:Maxstep
    step
    %记录gbest的适应度值
    f_gbest_his(step)=f_gbest;  %包含了初始f_gbest，但是没有最后一次迭代的f_gbest
    %线性下降收缩扩展系数
    %情况1：固定
    b=0.8;     %取固定收缩扩展系数
    %情况2：线性下降
    b = 0.368+(1.781-0.368)*(Maxstep-step)/Maxstep;    %b为收缩-扩张系数，从1线性下降到0.5
    %情况3：自适应
    b=0.368+(1.781-0.368)*(1/(1+exp(20*(step/Maxstep-0.5))));
    b_his(step)=b;             %记录b的值
    mbest =sum(pbest,3)/PopNum;   %mbest为平均最好位置
    dw=0.6+(1.2-0.6)*(1/(1+exp(20*(step/Maxstep-0.5))));
    cf=0.01+(0.25-0.01)*(1/(1+exp(20*(step/Maxstep-0.5))));
    for i = 1:PopNum  %PopNum为种群规模
        %位置更新
        eta=exp(-(pop(:,1:pop_RuleNum(i),i)-gbest(:,1:pop_RuleNum(i))).^2/(dw).^2); %吸引子自适应调整
        a = rand(2*InDim,pop_RuleNum(i)); u = rand(2*InDim,pop_RuleNum(i));  %a和u为二维矩阵
        p =eta.* a.*pbest(:,1:pop_RuleNum(i),i)+(1-eta).*(1-a).*gbest(:,1:pop_RuleNum(i));
        pop(:,1:pop_RuleNum(i),i) = p + b*abs(mbest(:,1:pop_RuleNum(i))-pop(:,1:pop_RuleNum(i),i)).*...
            log(1./u).*(1-2*(u >= 0.5));       
        %边界检查
        for r=1:InDim   %中心检查
            for j=1:pop_RuleNum(i)
                if pop(r,j,i)<pop_bound_center(1)
                    pop(r,j,i)=pop_bound_center(1).*(1+cf.*rand);
                end
                if pop(r,j,i)>pop_bound_center(2)
                    pop(r,j,i)=pop_bound_center(2).*(1-cf.*rand);
                end
            end
        end
        for r=InDim+1:2*InDim   %宽度检查
            for j=1:pop_RuleNum(i)
                if pop(r,j,i)<pop_bound_width(1)
                    pop(r,j,i)=pop_bound_width(1).*(1+cf.*rand);
                end
                if pop(r,j,i)>pop_bound_width(2)
                    pop(r,j,i)=pop_bound_width(2).*(1-cf.*rand);
                end
            end
        end    
        %适应度计算，并估计权重
        [fit(i),Weights(i).Weights]= fitness(pop(:,:,i),pop_RuleNum(i),TrainSamIn,TrainSamOut);
        %个体最优位置pbset更新
        if fit(i) < f_pbest(i)
            pbest(:,:,i) = pop(:,:,i);  %决策空间更新
            f_pbest(i) = fit(i);    %目标空间更新
        end
        %全局最优位置gbest更新
        if f_pbest(i) < f_gbest
            gbest = pbest(:,:,i);    %决策空间更新
            f_gbest = f_pbest(i);  %目标空间更新
            Weights_best= Weights(i).Weights;
            RuleNum_bset=pop_RuleNum(i);
        end
    end 
    %记录最佳的模糊规则数
    RuleNum_best_his=[RuleNum_best_his RuleNum_bset];
end

%% 解出中心，宽度和权值
Center=gbest(1:InDim,1:RuleNum_bset);        %前40个元素是中心
Width=gbest(InDim+1:2*InDim,1:RuleNum_bset);     %后40个元素是宽度
Weights=Weights_best;

%% 训练集预测
NormValueMatrix=[]; %清空，以便训练集预测
RegressorMatrix=[];%清空，以便测试集预测
NormValueMatrix=GetMeNormValue(TrainSamIn,Center,Width);%计算全体训练样本的规则层输出RuleUnitOut
RegressorMatrix=GetMeRegressorMatrix(NormValueMatrix,TrainSamIn);%计算全体训练样本的回归量RegressorMatrix
TrainNetOut=Weights*RegressorMatrix;%NetOut为网络输出
TrainNetOutN=mapminmax('reverse',TrainNetOut,outputps); %训练集输出反归一化

%% 计算训练集RMSE、APE、精度
TrainError=TrainSamOutN-TrainNetOutN;
TrainRMSE=sqrt(sum(TrainError.^2)/TrainSamNum);
TrainAPE=sum(abs(TrainError)./abs(TrainSamOutN))/TrainSamNum;
TrainAccuracy=sum(1-abs(TrainError./TrainSamOutN))/TrainSamNum;
disp(['TrainRMSE     == ',num2str(TrainRMSE),' ']); %训练RMSE
disp(['TrainAPE      == ',num2str(TrainAPE),' ']); %训练APE
disp(['TrainAccuracy == ',num2str(TrainAccuracy),' ']); %训练精度

%% 测试集预测
NormValueMatrix=[]; %清空，以便测试集预测
RegressorMatrix=[]; %清空，以便测试集预测
%首先计算窗口内样本的规则层输出阵NormValueMatrix
NormValueMatrix=GetMeNormValue(TestSamIn,Center,Width);
%根据规则层输出NormValue，得到回归量矩阵RegressorMatrix
RegressorMatrix=GetMeRegressorMatrix(NormValueMatrix,TestSamIn); %RegressorMatrix是M行N列，M=RuleNum*(InDim+1)，N是窗口内样本数
TestNetOut=Weights*RegressorMatrix;%NetOut为网络输出
TestNetOutN=mapminmax('reverse',TestNetOut,outputps); %训练集输出反归一化

%% 计算测试集RMSE、APE、精度
TestError=TestSamOutN-TestNetOutN;
TestRMSE=sqrt(sum(TestError.^2)/TestSamNum);
TestAPE=sum(abs(TestError)./abs(TestSamOutN))/TestSamNum;
TestAccuracy=sum(1-abs(TestError./TestSamOutN))/TestSamNum;
disp(['TestRMSE      == ',num2str(TestRMSE),' ']); %测试RMSE
disp(['TestAPE       == ',num2str(TestAPE),' ']); %测试APE
disp(['TestAccuracy  == ',num2str(TestAccuracy),' ']); %测试精度

%% 绘图
figure; %收缩扩张系数变化趋势
plot(b_his,'k-','LineWidth',2);
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉

figure;  %模糊规则数
plot(RuleNum_best_his,'k-','LineWidth',2);
xlabel('Number of iterations','fontsize',10,'fontname','Times New Roman')
ylabel('Number of fuzzy rules','fontsize',10,'fontname','Times New Roman')
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.16 .16 .80 .74]);  %调整 XLABLE和YLABLE不会被切掉
ylim([0 16])

figure;  %最佳适应度值进化曲线
plot(f_gbest_his,'k-','LineWidth',2);
xlabel('Number of iterations','fontsize',10,'fontname','Times New Roman')
ylabel('\itf\iti\itt','fontsize',10,'fontname','Times New Roman')
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.18 .16 .78 .74]);  %调整 XLABLE和YLABLE不会被切掉

%训练结果作图
figure;
plot(TrainSamOutN,'k-','LineWidth',2)
hold on
plot(TrainNetOutN,'r--','LineWidth',2)
h=legend('Real values','Forecasting output');
set(h,'Box','off','Fontsize',10,'fontname','Times New Roman');
xlabel('Training samples','fontsize',10,'fontname','Times New Roman')
ylabel('Desired and predicted outputs','fontsize',10,'fontname','Times New Roman')
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试结果作图
figure;
kk=TrainSamNum+1:TrainSamNum+TestSamNum;
plot(kk,TestSamOutN,'k-','LineWidth',2)
hold on
plot(kk,TestNetOutN,'r--.','LineWidth',2,'Markersize',5)
h=legend('Desired output','Predicted output');
set(h,'Box','off','Fontsize',10,'fontname','Times New Roman','location','northeast');
xlabel('Testing samples','fontsize',10,'fontname','Times New Roman')
ylabel('Outputs','fontsize',10,'fontname','Times New Roman')
ylim([0.2 1.6])
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.14 .16 .80 .80]);  %调整 XLABLE和YLABLE不会被切掉

%测试数据误差
figure;
kk=TrainSamNum+1:TrainSamNum+TestSamNum;
plot(kk,TestError,'k-','LineWidth',2)
xlabel('Testing samples','fontsize',10,'fontname','Times New Roman')
ylabel('Prediction error','fontsize',10,'fontname','Times New Roman')
set(gcf,'Position',[100 100 320 250]);
set(gca,'Position',[.17 .16 .79 .80]);  %调整 XLABLE和YLABLE不会被切掉

%% 存储结果
save RuleNum_best_his RuleNum_best_his
save f_gbest_his f_gbest_his
save TestSamOutN TestSamOutN
save TestNetOutN TestNetOutN
save TestError TestError
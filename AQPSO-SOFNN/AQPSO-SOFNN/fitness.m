function [fit,Weights]=fitness(pop,RuleNum,TrainSamIn,TrainSamOut)
% 输入
[InDim,TrainSamNum]=size(TrainSamIn); %输入维数、训练样本数
OutDim=size(TrainSamOut,1);   %输出维数

Center=[];Width=[];NormValueMatrix=[];RegressorMatrix=[];Weights=[];

Center=pop(1:InDim,1:RuleNum); %前4行元素是中心
Width=pop(InDim+1:2*InDim,1:RuleNum);        %后4行元素是宽度

% 首先计算窗口内样本的规则层输出阵NormValueMatrix
NormValueMatrix=GetMeNormValue(TrainSamIn,Center,Width);
% 根据规则层输出NormValue，得到回归量矩阵RegressorMatrix
RegressorMatrix=GetMeRegressorMatrix(NormValueMatrix,TrainSamIn); %RegressorMatrix是M行N列，M=RuleNum*(InDim+1)，N是窗口内样本数


% 根据回归量矩阵RegressorMatrix，得到Hermitian矩阵H,结果参数(输出权值)Weights
Weights=DeriveWeights(RegressorMatrix,TrainSamOut);   %10条模糊规则里面有50个权重系数
NetOut=Weights*RegressorMatrix;
% 计算训练样本的RMSE
RMSE=sqrt(sumsqr(TrainSamOut-NetOut)/(OutDim*TrainSamNum)); %训练样本的RMSE

fit=RMSE*(1+0.9*RuleNum); %均方根误差就是适应度值
end
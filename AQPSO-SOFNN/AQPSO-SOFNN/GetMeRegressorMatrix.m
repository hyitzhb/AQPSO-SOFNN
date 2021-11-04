function RegressorMatrix=GetMeRegressorMatrix(NormValue,SamIn)
% 这个子程序是用来产生一个矩阵，以便计算模糊规则的结果参数

%RuleUnitOut---RBF单元的输出或规则层神经元的输出
%SamIn-- TrainSamIn(:,k)，即训练样本输入，可能是单个样本，也可能是到目前为止的训练样本集
%对于单个样本来说，有几个模糊规则数，RuleUnitOut就为RuleNum*1
%对于多个样本来说，有几个模糊规则数，RuleUnitOut就为RuleNum*SamNum

[RuleNum,SamNum]=size(NormValue); %RuleNum为规则数，即RuleUnitOut的行数代表规则数
[InDim,SamNum]=size(SamIn); %SamNum为样本数，即RuleUnitOut的列数代表样本数

for j=1:SamNum
    for i=1:InDim
        PA((i-1)*RuleNum+1 : i*RuleNum,j )=SamIn(i,j)*NormValue(:,j);
    end
end

RegressorMatrix=[NormValue;PA];

function NormValue=GetMeNormValue(TrainSamIn,Center,Width)
% 功能：计算规则层输出
% 输入：
%     TrainSamIn_All：训练样本输入， InDim*SamNum
%     Center: 中心，InDim*RuleNum
%     Width: 宽度，InDim*RuleNum
%     Width_Left: 左宽度，InDim*RuleNum
%     Width_Right: 右宽度，InDim*RuleNum
% 输出：
% RuleUnitOut: RuleNum行SamNum列

[InDim,SamNum]=size(TrainSamIn); %InDim是输入维数，SamNum是样本个数
[InDim,RuleNum]=size(Center); %InDim是输入维数，RuleNum是模糊规则数

for k=1:SamNum   %多少个样本就循环多少次
    %     k=1 %测试用
    SamIn=TrainSamIn(:,k);
    
    %隶属函数层输出
    for i=1:InDim
        for j=1:RuleNum

            MemFunUnitOut(i,j)=exp(-(SamIn(i)-Center(i,j))^2/Width(i,j)^2);
        end
    end
    % 规则层
    RuleUnitOut(:,k)=prod(MemFunUnitOut,1); %规则层输出,行数代表规则层各节点的输出，列数代表对应的样本
   
    % 归一化层
    RuleUnitOutSum(k)=sum(RuleUnitOut(:,k)); %规则层输出求和
    NormValue(:,k)=RuleUnitOut(:,k)./RuleUnitOutSum(k); %归一化层输出，自组织调整NormValue
end





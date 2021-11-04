

function Weights=DeriveWeights(RegressorMatrix,TrainSamOut_Window)

% 功能： 根据窗口内数据，利用伪逆技术，辨识出结果参数(输出权值)Weights以及神经网络输出NetOut

% 输入：
% TrainSamIn_Window：当前窗口内的训练样本的输入
% Center：中心
% Width_Left：左宽度
% Width_Right：右宽度
% TrainSamOut_Window：当前窗口内的训练样本的输出

% 输出：
% Q:Hermitian矩阵
% Weights:模糊规则的后件参数，输出权值
% NetOut:当前窗口内样本的神经网络输出

Q = pinv(RegressorMatrix*RegressorMatrix'); %Q是Hermitian矩阵,Q是M*M，即M=RuleNum*(InDim+1)
Weights=(Q*RegressorMatrix*TrainSamOut_Window')'; %TrainSamOut_Window=1行N列，Weights是1行M列典型的线性最小二乘法

end

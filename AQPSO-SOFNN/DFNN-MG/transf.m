%计算结果参数？回归最小二乘？

function BB=transf(A,P)
%A---output of RBF  A是规则层的输出(已经规范化),一个规则的时候为1*1
%P-- sample of input P是当前样本的输入 4*1

[u,N]=size(A);  %u为当前规则数，N为
[r,N]=size(P);  %r为输入样本的维数，N为输入样本的数目（可能为1或当前所有训练数据）
for j=1:N
   for i=1:r
      PA((i-1)*u+1:i*u,j)=P(i,j)*A(:,j); %TSK模型？
   end 
end
BB=[A;PA];

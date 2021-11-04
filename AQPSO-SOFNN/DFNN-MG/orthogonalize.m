function [w,a]=orthogonalize(p)
%正交化计算，计算宽度
% This program is used to transform p into orthogonalized w

[u,v]=size(p);
w(:,1)=p(:,1);
a=eye(v);
for k=2:v
b=zeros(u,1);
   for i=1:k-1
      a(i,k)=w(:,i)'*p(:,k)/(w(:,i)'*w(:,i));
      b=b+a(i,k)*w(:,i);
   end
      w(:,k)=p(:,k)-b;
end
   


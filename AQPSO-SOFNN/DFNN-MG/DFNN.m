function [w1,w2,width,rule,e,RMSE]=DFNN(p,t,parameters)
% This is D-FNN training program.
% Input:
%     p is the input data,which is r by q matrix
%       r is NO. of input
%     t is the output data, which is s2 by q matrix.
%       q is the NO. of sample data.
%     parameters is a vector which defines the predefined parameters
%     parameters(1)=kdmax  (max of accommodation criterion) 调节kd允许的最大值
%     parameters(2)=kdmin  (min of accommodation criterion) 调节kd允许的最小值
%     parameters(3)=gama  (decay constant) 衰减常数，调节kd
%     parameters(4)=emax   (max of output error) 定义的最大误差
%     parameters(5)=emin   (min of output error) 定义的最小误差
%     parameters(6)=beta  (convergence constant) 误差的收敛常数
%     parameters(7)=width0  (the width of the first rule) 第一条模糊规则的宽度
%     parameters(8)=k   (overlap factor if RBF units) 重叠因子
%     parameters(9)=kw   (width updating factor) 宽度调整常数
%     parameters(10)=kerr  (significance of a rule) 修剪规则用的预设阈值
% Output:
%     w1 is the centers of RBF units,which is a u by r matrix  中心
%     w2 is the weights,which is s2 by u(1+r) matrix 权值
%     width is the widths fo RBF units,which is a 1 by u matrix  宽度


[r,q]=size(p);  %p是训练集输入，r是输入维数，q是训练集样本数目
[s2,q]=size(t); %t是
%setting predefined parameters
kdmax=parameters(1);
kdmin=parameters(2);
gama=parameters(3);
emax=parameters(4);
emin=parameters(5);
beta=parameters(6);
width0=parameters(7);
k=parameters(8);
kw=parameters(9);
kerr=parameters(10);
ALLIN=[]; ALLOUT=[]; CRBF=[];
%when first sample data coming
ALLIN=p(:,1); ALLOUT=t(:,1);
%Setting up the initial DFNN
CRBF=p(:,1);w1=CRBF';width(1)=width0;rule(1)=1;
%Caculating the first out error
a0=RBF(dist(w1,ALLIN),1./width');
a01=[a0 p(:,1)'];
w2=ALLOUT/a01';
a02=w2*a01';
RMSE(1)=sqrt(sumsqr(ALLOUT-a02)/s2);
%when other sample data coming
for i=2:q
%     i=2;
    IN=p(:,i); OUT=t(:,i);
    ALLIN=[ALLIN IN];
    ALLOUT=[ALLOUT OUT];
    [r,N]=size(ALLOUT); 
    [s r]=size(w1);
    dd=dist(w1,IN);
    [d,ind]=min(dd);
    kd=max(kdmax*gama.^(i-1),kdmin);
    %Caculating the actual output if ith sample data
    ai=RBF(dist(w1,IN),1./width');
    ai=ai/sum(ai);ai1=transf(ai,IN);
    ai2=w2*ai1;
    errout=t(:,i)-ai2;
    errout=sum(errout.*errout)/s2;
    e(i)=sqrt(errout);
    ke=max(emax*beta.^(i-1),emin);
    if d>kd
       if e(i)>ke
           CRBF=[CRBF IN]; %add a new rule
           wb=k*d;
           width=[width wb];
           w1=CRBF';
           [u,v]=size(w1);
           %Caculating outputs of RBF after growing for all data
           A=RBF(dist(w1,ALLIN),1./width');
           A0=sum(A);
           A1=A./(ones(u,1)*A0);
           A2=transf(A1,ALLIN);
           if u*(r+1)<=N
               %caculating error reduction rate
               tT=ALLOUT';
               PAT=A2';
               [W,AW]=orthogonalize(PAT);
               SSW=sum(W.*W)';SStT=sum(tT.*tT)';
               err=((W'*tT)'.^2)./(SStT*SSW');
               errT=err';
               err1=zeros(u,s2*(r+1));
               err1(:)=errT;
               err21=err1';
               err22=sum(err21.*err21)/(s2*(r+1));
               err23=sqrt(err22);
               No=find(err23<kerr);
               if ~isempty(No)
                   CRBF(:,No)=[];w1(No,:)=[];
                   width(:,No)=[];err21(:,No)=[];
                   [uu,vv]=size(w1);
                   AA=RBF(dist(w1,ALLIN),1./width');
                   AA0=sum(AA);
                   AA1=AA./(ones(uu,1)*AA0);
                   AA2=transf(AA1,ALLIN);
                   w2=ALLOUT/AA2;
                   outAA2=w2*AA2;
                   sse0=sumsqr(ALLOUT-outAA2)/(i*s2);
                   RMSE(i)=sqrt(sse0);
                   rule(i)=uu;
                   w2T=w2';ww2=zeros(uu,s2*(r+1));
                   ww2(:)=w2T;
                   w21=ww2';
               else
                   w2=ALLOUT/A2;
                   outA2=w2*A2;
                   sse0=sumsqr(ALLOUT-outA2)/(s2*i);
                   RMSE(i)=sqrt(sse0);
                   rule(i)=u;
                   w2T=w2';ww2=zeros(u,s2*(r+1));
                   ww2(:)=w2T;
                   w21=ww2';
               end
           else
               w2=ALLOUT/A2;
               outA2=w2*A2;
               sse0=sumsqr(ALLOUT-outA2)/(s2*i);
               RMSE(i)=sqrt(sse0);
               rule(i)=u;
               w2T=w2';ww2=zeros(u,s2*(r+1));
               ww2(:)=w2T;
               w21=ww2';
           end
       else
           a=RBF(dist(w1,ALLIN),1./width');
           a0=sum(a);a1=a./(ones(s,1)*a0);
           a2=transf(a1,ALLIN);
           w2=ALLOUT/a2;
           outa2=w2*a2;
           sse1=sumsqr(ALLOUT-outa2)/(s2*i);
           RMSE(i)=sqrt(sse1);
           rule(i)=s;
       end
    else
        if e(i)>ke
            width(ind)=kw*width(ind);
            aa=RBF(dist(w1,ALLIN),1./width');
            aa0=sum(aa);aa1=aa./(ones(s,1)*aa0);
            aa2=transf(aa1,ALLIN);
            w2=ALLOUT/aa2;
            outaa2=w2*aa2;
            sse2=sumsqr(ALLOUT-outaa2)/(i*s2);
            RMSE(i)=sqrt(sse2);
            rule(i)=s;
        else
            aa1=RBF(dist(w1,ALLIN),1./width');
            aa01=sum(aa1);aa11=aa1./(ones(s,1)*aa01);
            aa21=transf(aa11,ALLIN);
            w2=ALLOUT/aa21;
            outaa21=w2*aa21;
            sse3=sumsqr(ALLOUT-outaa21)/(s2*i);
            RMSE(i)=sqrt(sse3);
            rule(i)=s;
        end
    end
end
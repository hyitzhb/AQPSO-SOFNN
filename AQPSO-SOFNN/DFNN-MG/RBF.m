function a=RBF(dis,width)
% This program is used to calculate the RBF units.
% Input:
%   d is the distance matrix
%   w is the width matrix ¿í¶È¾ØÕó
% Output:
%   a is the output of RBF units
% a ÊÇRBF²ãÊä³ö

dis=dis.*width(:,ones(1,size(dis,2)));
a=exp(-(dis.*dis));
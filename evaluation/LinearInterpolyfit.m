function [ val ] = LinearInterpolyfit( dx,dy,v )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
dx =[0;dx];
dy =[dy(1);dy];


lenx = length(dx);
leny = length(dy);
if lenx~=leny
    error(['length is not equal']);
end
if v == 0
    val = dy(1);
    return;
end
leftindex = 1;
rightindex = lenx;
mid = floor((lenx+1)/2);
while rightindex>leftindex+1
    if v>dx(mid)
        leftindex = mid;        
    else
        rightindex = mid;
    end
    mid = floor((leftindex+rightindex)/2);    
end
if abs(dx(rightindex)-dx(leftindex))<0.001
    val = dy(leftindex);
else
    val = dy(leftindex)+(v-dx(leftindex))/(dx(rightindex)-dx(leftindex))*(dy(rightindex)-dy(leftindex));
end
end
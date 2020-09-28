function [ labelstruct ] = LabelAnalysis( labelfile )
%UNTITLED4 Analysis Label file
%   Input:Labelfile (id,catagoryid)
%   Output:LabelStruct
%   label: Input labelfile
%   labelnum: total Number of catagory
%   categorynum: Number of label in each catagory
label = dlmread(labelfile);
labelstruct.label = label;
labelstruct.labelnum = max(label(:,2));
labelstruct.categorynum = zeros(labelstruct.labelnum,1);
for i = 1 : labelstruct.labelnum
    id = find(label(:,2)==i);
    labelstruct.categorynum(i) = length(id);
end
end


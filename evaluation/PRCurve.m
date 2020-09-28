function [precision, recall ] = PRCurve( modelid, returnlist, labelinfo,labelnum)
%UNTITLED2 Summary of this function goes here
%   modelid: the id of the model == GroundTruth
%   returnlist: the ordered models == Retrieval Results
%   labelinfo: catagory of each model
%   labelnum: total number of labels

categoryid = labelinfo(modelid); % catagory of Ground Truth model
ln = length(returnlist); % number of retrievaled models
precision = zeros(ln,1);
recall = zeros(ln,1);
corrid = 0; %number of correct ID
for i = 1 : ln
    if labelinfo(returnlist(i)) == categoryid
        corrid = corrid + 1;
        precision(corrid) = corrid/i; % precision decreases with i increasing
        recall(corrid) = corrid/labelnum;% recall increase with i increasing
        if corrid == labelnum
            break;
        end
    end
end
precision = precision(1:corrid);
recall = recall(1:corrid);
end


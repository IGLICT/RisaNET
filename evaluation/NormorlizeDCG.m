function [ndcg, dcg ] = NormorlizeDCG( modelid, returnlist, labelinfo,labelnum)
%   Return the DCG value of each input model
%   modelid: the id of the model
%   returnlist: the ordered models 
%   labelinfo
categoryid = labelinfo(modelid);
ln = length(returnlist);
corrid = 0;
dcglist = zeros(ln,1);
knum = 0;

for i = 1 : ln
    if labelinfo(returnlist(i)) == categoryid
    %    corrid = corrid + 1;
        dcglist(i) = 1;
    %    if corrid == labelnum
    %        knum = i;
    %        break;
    %    end
    end
end
dcg = dcglist(1);
for i = 2 : ln
    dcg = dcg+dcglist(i)/log2(i);
end
mp = 1;
for i = 2 : labelnum
    %dcg = dcg+dcglist(i)/log2(i);
    mp = mp+1/log2(i);
end
ndcg = dcg/mp;
end


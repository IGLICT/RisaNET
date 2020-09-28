function [FirstTier, SecondTier, Nearest ] = Tier( modelid, returnlist, labelinfo,labelnum)
%	return firstTier and secondTier of each input model
%   modelid: the id of the model
%   returnlist: the ordered models 
%   labelinfo: catagory of each model
%   labelnum: number of models in the correct catagory

categoryid = labelinfo(modelid);
ln = length(returnlist);
corrid = 0;
FirstTier = 0; 
SecondTier = 0;
Nearest = 0;
ftlen = labelnum;
stlen = 2*ftlen;
tick = min(stlen, ln);

if labelinfo(returnlist(1)) == categoryid
    Nearest = 1;
end

for i = 1 : tick
    if labelinfo(returnlist(i)) == categoryid
        corrid = corrid + 1;
    end
    if i == ftlen
        FirstTier = corrid/labelnum;            
    end
    if i == stlen
        SecondTier = corrid/labelnum;
    end
    if corrid == labelnum
        %break;
    end

end
end


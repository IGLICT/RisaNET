function [ st,prv,retrievalindex ] = IDPRCurve_k1( idmatname, labelname ,method)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%   read label

labelstruct = LabelAnalysis(labelname);
dismat = load(idmatname);
%%%%%change model name when necessary%%%%
% dismat = dismat.ans;
dismat = dismat.ids1;
% dismat = dismat.idmat;
[m,n] = size(dismat);
% m: number of total models
modelnum = m;
% modelnum: number of total models
labelinfo = zeros(m,1);
%   labelinfo: catagory of each returned model
for i = 1 : size(labelstruct.label,1)
    labelinfo(labelstruct.label(i,1)) = labelstruct.label(i,2);
end
%   assign all label with correct directory

pr = cell(m,1);
for i = 1 : modelnum
    modelid = labelstruct.label(i,1);
    ids = dismat(i,:);
    iid = find(ids == modelid);
    nid = find(ids~=modelid);
    returnid = ids(nid);
%     zeroid = find(returnid == 0);
%     nzeroid = find(returnid~=0);
%     returnid = returnid(nzeroid);
%     if iid~=1 && iid~=n
%         returnid =ids([1:iid-1,iid+1:n]);
%     end
%     if iid == 1
%         returnid =ids(2:n);
%     end
%     if iid == n
%         returnid = ids(1:n-1);
%     end
    
    %returnid = ids;
    [pr{i}.precision, pr{i}.recall] = PRCurve( modelid, returnid, labelinfo,labelstruct.categorynum(labelstruct.label(i,2))-1);
    [pr{i}.firsttier, pr{i}.secondtier, pr{i}.nearest ] = Tier( modelid, returnid, labelinfo,labelstruct.categorynum(labelstruct.label(i,2))-1);
    [pr{i}.ndcg, pr{i}.dcg ] = NormorlizeDCG( modelid, returnid, labelinfo,labelstruct.categorynum(labelstruct.label(i,2))-1);
    
end
% for each model, search its nearest models and calculate PRCurve index
macro_ft = zeros(labelstruct.labelnum, 1);
macro_st = zeros(labelstruct.labelnum, 1);
macro_map = zeros(labelstruct.labelnum, 1);
macro_ndcg = zeros(labelstruct.labelnum, 1);
macro_nearest = zeros(labelstruct.labelnum, 1);

ft = 0;
st = 0;
ndcg = 0;
nearest = 0;
%dcg = 0;
map = 0;

for i = 1 : modelnum
    map = map + mean(pr{i}.precision);
    macro_map(labelstruct.label(i,2)) = macro_map(labelstruct.label(i,2)) + mean(pr{i}.precision);
    ft= ft+pr{i}.firsttier;
    macro_ft(labelstruct.label(i,2)) = macro_ft(labelstruct.label(i,2)) + pr{i}.firsttier;
    st = st + pr{i}.secondtier;
    macro_st(labelstruct.label(i,2)) = macro_st(labelstruct.label(i,2)) + pr{i}.secondtier;
    ndcg = ndcg + pr{i}.ndcg;
    macro_ndcg(labelstruct.label(i,2)) = macro_ndcg(labelstruct.label(i,2)) + pr{i}.ndcg;
    nearest = nearest + pr{i}.nearest;
    macro_nearest(labelstruct.label(i,2)) = macro_nearest(labelstruct.label(i,2)) + pr{i}.nearest;
%     dcg = dcg + pr{i}.dcg;    
end
for c = 1:labelstruct.labelnum
    macro_ft(c) = macro_ft(c)/labelstruct.categorynum(c);
    macro_st(c) = macro_st(c)/labelstruct.categorynum(c);
    macro_map(c) = macro_map(c)/labelstruct.categorynum(c);
    macro_ndcg(c) = macro_ndcg(c)/labelstruct.categorynum(c);
    macro_nearest(c) = macro_nearest(c)/labelstruct.categorynum(c);
end
    
retrievalindex.firsttier = ft/modelnum;
retrievalindex.secondtier = st/modelnum;
retrievalindex.ndcg = ndcg/modelnum;
retrievalindex.nearest = nearest/modelnum;
retrievalindex.map = map/modelnum;
% retrievalindex.dcg = dcg/modelnum;
retrievalindex.macro_ft = mean(macro_ft);
retrievalindex.macro_st = mean(macro_st);
retrievalindex.macro_map = mean(macro_map);
retrievalindex.macro_ndcg = mean(macro_ndcg);
retrievalindex.macro_nearest = mean(macro_nearest);

t = 0.1: 0.01 : 1;
v = zeros(length(t),1);
for ti = 1:length(t)    
    for i = 1 : modelnum
        if ~isempty(pr{i}.recall)
            [ val ] = LinearInterpolyfit( pr{i}.recall,pr{i}.precision,t(ti) );
            v(ti) = v(ti)+val;
        end
    end
end
v = v/modelnum;
%plot(t,v);
st = t;
prv = v;


%hold all;
figure
str1 = '#0087F9';
str2 = '#FFEA0E';
str3 = '#672BAF';
str4 = '#F68E00';
str5 = '#F83699';
str6 = '#05A203';
str7 = '#DD4837';
color1 = sscanf(str1(2:end),'%2x%2x%2x',[1 3])/255;
color2 = sscanf(str2(2:end),'%2x%2x%2x',[1 3])/255;
color3 = sscanf(str3(2:end),'%2x%2x%2x',[1 3])/255;
color4 = sscanf(str4(2:end),'%2x%2x%2x',[1 3])/255;
color5 = sscanf(str5(2:end),'%2x%2x%2x',[1 3])/255;
color6 = sscanf(str6(2:end),'%2x%2x%2x',[1 3])/255;
color7 = sscanf(str7(2:end),'%2x%2x%2x',[1 3])/255;
hold on;
if strcmp(method, 'a') == 1
    plot(st,prv,'Color', color1,'LineWidth',4);
end
if strcmp(method, 'b') == 1
    plot(st,prv,'Color', color2,'LineWidth',4);
end
if strcmp(method, 'c') == 1
    plot(st,prv,'Color', color3,'LineWidth',4);
end
if strcmp(method, 'd') == 1
    plot(st,prv,'Color', color4,'LineWidth',4);
end
if strcmp(method, 'e') == 1
    plot(st,prv,'Color', color5,'LineWidth',4);
end
if strcmp(method, 'f') == 1
    plot(st,prv,'Color', color6,'LineWidth',4);
end
if strcmp(method, 'g') == 1
    plot(st,prv,'Color', color7,'LineWidth',4);
end

xlabel('Recall')
ylabel('Precision')
%plot(st,prv,'-b','LineWidth',3);
%plot(st,prv,'-c','LineWidth',3);
%plot(st,prv,'-y','LineWidth',3);

% matname = [method, 'RetrieveIdx.mat'];
% save(matname,'st','prv','retrievalindex')
end


function [edge_feature, neighbour] = get_tpami_fv_dege(obj_folder, total_num)
% total_num = [1, max]

GetFeatureEA_new(obj_folder);
Angle =dlmread([obj_folder,'\angle_val.txt']);
Length =dlmread([obj_folder,'\edge_length.txt']);
Length = Length * 1000;

strnam = dir([obj_folder, '\*.obj']);
[modelnum, ~] = size(strnam);

[vsimpmesh,~,~,~,~,VVsimp,~,~,~,b_E,e2v] = cotlpml([obj_folder,'\','0.obj']);
neighbour = [b_E(:, 2), b_E(:, 4), b_E(:, 3), b_E(:, 5)];

%% Modified fms and fmlogdr
objlist=dir([obj_folder,'\*.obj']);
[~,i]=sort_nat({objlist.name});
objlist=objlist(i);
maxvalid = str2num(erase(objlist(length(objlist)).name, '.obj'));
[s1, s2] = size(Angle);
edge_feature = zeros(total_num, s2, 2);
for i = 1 : s1
    id = str2num(erase(objlist(i).name,'.obj'));
    if id ~= 0
        edge_feature(id, :, 1) = Angle(i, :);
        edge_feature(id, :, 2) = Length(i, :);
    end
end

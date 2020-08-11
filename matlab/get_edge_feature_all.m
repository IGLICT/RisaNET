main_folder = 'path\to\objs'; %
part_num = 25; % number of shape parts
model_num = 2611; % number of shapes
for i = 1:part_num
    input_folder =[ main_folder, '\p',num2str(i)];
    [edge_feature, e_neighbour] = get_tpami_fv_dege(input_folder, model_num);
    if i==1
        [ ~, e_num,  ~] = size(edge_feature);
        A_edge_feature = zeros(part_num, model_num, e_num, 2);
    end
    A_edge_feature(i,:,:,:) = edge_feature;
end
edgefeature = A_edge_feature;
if ~isempty(find(isnan(edgefeature), 1))
    display(find(isnan(edgefeature)));
end

mat_name = fullfile(main_folder, 'edgefeature.mat');
save(mat_name, 'edgefeature', 'e_neighbour', '-v7.3');
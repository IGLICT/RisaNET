label_path = '../pre_processed_labels/label_guitar';
data_path = '../example_checkpoints';
letter = 'a';
method = 0; 
%---Data 
data = h5read([data_path, '\0test_index.h5'],'/feature_vector');
data = data';

%---Label  %0-rimd;1-lf/shd; 2-mvcnn; 3-rotation; 4-sphere; 5-seqview;
labelstruct = LabelAnalysis( [label_path,'\label2.label'] );
tr_num = find(labelstruct.label(:, 2)==1);
test_num = find(labelstruct.label(:, 2)==2);
tr_szdata = length(tr_num);
te_szdata = length(test_num);
al_label = '\label1.label';
tr_data = data(tr_num, :);
test_data = data(test_num, :);
al_label = '\label1.label';
te_label = '\label_te.label';
tr_label = '\label_tr.label';

% -----All %0-rimd;1-lf/shd; 2-mvcnn; 3-rotation; 4-sphere; 5-seqview;
[szdata,~ ] = size(data);
ids2 = knnsearch(data, data,'k',szdata);
save('id2.mat','ids2');
[~,~,rtvidx_all] = IDPRCurve_k2('id2.mat',[label_path, al_label],'a');
ids2 = knnsearch(test_data, test_data,'k',te_szdata);
save('id2.mat','ids2');
[~,~,rtvidx_te] = IDPRCurve_k2('id2.mat',[label_path, te_label],letter);
 ids2 = knnsearch(tr_data, tr_data,'k',tr_szdata);
save('id2.mat','ids2');
[~,~,rtvidx_tr] = IDPRCurve_k2('id2.mat',[label_path, tr_label],'a');   
% rtv_tr_data = [rtvidx_tr.nearest, rtvidx_tr.firsttier, rtvidx_tr.secondtier, rtvidx_tr.ndcg, rtvidx_tr.map, rtvidx_tr.macro_nearest, rtvidx_tr.macro_ft, rtvidx_tr.macro_st, rtvidx_tr.macro_ndcg, rtvidx_tr.macro_map];
% rtv_te_data = [rtvidx_te.nearest, rtvidx_te.firsttier, rtvidx_te.secondtier, rtvidx_te.ndcg, rtvidx_te.map, rtvidx_te.macro_nearest, rtvidx_te.macro_ft, rtvidx_te.macro_st, rtvidx_te.macro_ndcg, rtvidx_te.macro_map];
% rtv_all_data = [rtvidx_all.nearest, rtvidx_all.firsttier, rtvidx_all.secondtier, rtvidx_all.ndcg, rtvidx_all.map, rtvidx_all.macro_nearest, rtvidx_all.macro_ft, rtvidx_all.macro_st, rtvidx_all.macro_ndcg, rtvidx_all.macro_map];
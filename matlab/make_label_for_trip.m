labelstruct = LabelAnalysis( '\\10.41.0.155\yangjiehome\furao\label\label_car\label1.label' );
% labelSeq = labelstruct.label(:,2);
% save('labelSeq.mat','labelSeq', '-v7.3');
labelMatrix = zeros(length(labelstruct.label));
for i = 1: length(labelstruct.label)
    for j = 1:length(labelstruct.label)
        if labelstruct.label(i,2)==labelstruct.label(j,2)
            labelMatrix(i,j)=1;
        else
            labelMatrix(i,j)=-1;
        end
    end
end
labelstruct = LabelAnalysis( '\\10.41.0.155\yangjiehome\furao\label\label_guitar\label2.label' );
train_idx = find(labelstruct.label(:, 2)==1);
test_idx = find(labelstruct.label(:, 2)==2);
save('Z:\carVAE\labelMatrix.mat','labelMatrix','train_idx', 'test_idx', '-v7.3');
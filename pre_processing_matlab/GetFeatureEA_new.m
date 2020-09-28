function [ ] = GetFeatureEA_new( srcfolder )
% calculate the base geometric feature of all shapes in the srcfolder,
% shapes in the source folder should ranked by natural number
cmdline = ['get_LAs.exe ',srcfolder];
dos(cmdline);
tarfvt1 = [srcfolder,'\edge_length.txt'];
tarfvt2 = [srcfolder,'\angle_val.txt'];
movefile('E:\data\edge_length.txt',tarfvt1,'f');
movefile('E:\data\angle_val.txt',tarfvt2,'f');
end
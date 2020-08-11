part_list = ["U:\RAOFU\DATA\3D\Car\process\p1", "U:\RAOFU\DATA\3D\Car\process\p2", "U:\RAOFU\DATA\3D\Car\process\p3", "U:\RAOFU\DATA\3D\Car\process\p4", "U:\RAOFU\DATA\3D\Car\process\p5", "U:\RAOFU\DATA\3D\Car\process\p6", "U:\RAOFU\DATA\3D\Car\process\p7"];
main_folder = 'U:\RAOFU\DATA\3D\Car\process\p1';
save_folder = 'U:\RAOFU\DATA\3D\Car\process';

obj_list = dir(fullfile(main_folder, '*.obj')); % This contains 0.obj
obj_num = length(obj_list)-1; %[0,num]
part_num = length(part_list);

structMatrix = zeros(obj_num, 8 * (part_num+1));
bodyinfo = zeros(obj_num, 12);
for i = 1: obj_num
    obj_name = [num2str(i),'.obj'];
    body = fullfile(main_folder,obj_name);
    structMatrix(i,8 * (part_num+1)) = 1;
    if isfile(body)
        [v1,~]=cotlpml(body);
        coeff = pca(v1);
        pc1 = coeff(:,1);
        pc2 = coeff(:,2);
        pc3 = coeff(:,3);
        bodycenter = mean(v1);
        bodyinfo(i,:) =[bodycenter, pc1', pc2', pc3'] ;
    end
end
pc1 = zeros(1,3);
pc2 = zeros(1,3);
pc3 = zeros(1,3);
for p = 1 : part_num
    head_dir = convertStringsToChars(part_list(p));
    for i = 1: obj_num
        obj_name = [num2str(i),'.obj'];
        part = fullfile(head_dir,obj_name);
        if isfile(part)
            [v1,~]=cotlpml(part);
            partcenter = mean(v1);
            coeff = pca(v1);
            ppc1 = coeff(:,1);
            ppc2 = coeff(:,2);
            ppc3 = coeff(:,3);
            bodycenter = bodyinfo(i,1:3);
            pc1 = bodyinfo(i,4:6);
            pc2 = bodyinfo(i,7:9);
            pc3 = bodyinfo(i,10:12);            
            t1 = dot(ppc1,pc1)/(norm(ppc1)*norm(pc1));
            t2 = dot(ppc1,pc2)/(norm(ppc1)*norm(pc2));
            t3 = dot(ppc1,pc3)/(norm(ppc1)*norm(pc3));
            t4 = dot(ppc2,pc1)/(norm(ppc2)*norm(pc1));
            t5 = dot(ppc2,pc2)/(norm(ppc2)*norm(pc2));
            t6 = dot(ppc2,pc3)/(norm(ppc2)*norm(pc3));
            a = bodycenter -partcenter;
            a = norm(a);
            structMatrix(i, p*8-7:p*8) = [a, t1, t2, t3, t4, t5, t6, 1];
        else
            structMatrix(i, p*8-7:p*8) = [0, 0, 0, 0, 0, 0, 0, 0];
        end
    end
end
save([save_folder, '\8_structMatrix.mat'],'structMatrix','-v7.3')

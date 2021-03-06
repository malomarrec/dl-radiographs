clear;
close all;
%%
r = 1:100; % size of the blob
number_blobs = [1];
pix_size = 0.1;
B0 = [0.1,10,20,30,40,50,60,70,80,90,100];
%B0 = [100,90,80,70,60,50,40,30,20,10,1,0.1]
Nproton = 1e7;
half_angle = 5; %degrees
i = 0;
run_nb = 100;
fileID = fopen('labels.txt'"', 'a');
for i1 = 1:max(size(r))
    for i2 = 1:max(size(number_blobs))
        for i3 = 1:max(size(B0))
            i = i+1
            name = sprintf('%s_%d_%d','workspace',run_nb,i);
            [y0_map,x0_map,final_Az, scale, xyz, dxyz, sign] = bmap_generator(r(i1),number_blobs(i2),pix_size,i);
            name_array_label = sprintf('workspaces/%s_%d_%d.mat','array_label',run_nb,i);
            save(name_array_label,'x0_map','y0_map','final_Az', 'scale')
%             h = figure;
%             set(gcf, 'Visible', 'off')
%             imagesc(y0_map,x0_map,final_Az*scale);
%             colorbar()
%             saveas(h,name_label)

            [edges_x,S2] = test_pusherv2(B0(i3), Nproton, half_angle, i);
            name_array_radio = sprintf('workspaces/%s_%d_%d.mat','array_radiograph',run_nb,i);
            save(name_array_radio,'edges_x','S2')
%             h2 = figure;
%             set(gcf, 'Visible', 'off')
%             imagesc(edges_x/1e4,edges_x/1e4,S2); colormap(flipud(colormap('gray')));
%             colorbar()
%             saveas(h2,name_radio)

            out = struct('run_nb',run_nb,'iteration', i,'r',i1,'nb_blobs',i2,'B0',i3,'Nproton',Nproton, 'xyz', xyz, 'dxyz',dxyz,'sign',sign);
            s = jsonencode(out);
            fwrite(fileID, s);
            
        end
    end
end
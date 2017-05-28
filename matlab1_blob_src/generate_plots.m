

%%Getmax


run_nbs =[1,2];
i_list = [39,607];

maximum_radio = 0
for run_nb = run_nbs
    for j = 1:i_list(run_nb)
    %     load(sprintf('%s_%d_%d.mat','array_label',run_nb,j))
    %     if (max(max(final_Az(:)), -min(final_Az(:))) > maximum_label)
    %         maximum_label = max(max(final_Az(:)), -min(final_Az(:)));
    %     end

        load(sprintf('workspaces/%s_%d_%d.mat','array_radiograph',run_nb,j))
        if (max(S2(:)) > maximum_radio)
            maximum_radio = max(S2(:));
        end
    %     max_label_name = sprintf('%s_%d.mat','max_label');
    %     save(max_label_name,'maximum_label')
        max_radio_name = sprintf('%s_%d.mat','max_radio');
        save(max_radio_name,'maximum_radio')
        maximum_radio
    end
end

%%Generate image
for run_nb = run_nbs
    for k = 1:i_list(run_nb)
%         load(sprintf('%s_%d_%d.mat','array_label',run_nb,k))
%         name_label = sprintf('%s_%d_%d.png','label',run_nb,k);
%         h_label = figure;
%         set(gcf, 'Visible', 'off')
%         imagesc(y0_map,x0_map,final_Az/maximum_label);colormap(flipud(colormap('gray')));
%         caxis([-1, 1])
%         axis equal;
%         saveas(h_label,name_label)

        load(sprintf('workspaces/%s_%d_%d.mat','array_radiograph',run_nb,k))
        name_radiograph = sprintf('%s_%d_%d.png','radiograph',run_nb,k);
        h_radiograph = figure;
        set(gcf, 'Visible', 'off')
        imagesc(edges_x/1e4,edges_x/1e4,S2/maximum_radio); colormap(flipud(colormap('gray')));
        caxis([0, 1])
        axis equal;
        saveas(h_radiograph,name_radiograph) 
    end
end





% 
% 
% %% Getting maxes
% 
% 
% maximum_radio = 0;
% maximum_label = 0;
% 
% 
% files = dir('*.mat');
% for file = files'
%     if(~(findstr(file.name,"array_radiograph") == []))
%         print("a")
% %         parsed = strsplit(file.name,"_");
% %         run_nb = str2num(parsed(2))
% %         k = str2num(parsed(3))
% %         pint(sprintf('%s_%d_%d.mat','array_radiograph',run_nb,k))
% %         %load(sprintf('%s_%d_%d.mat','array_radiograph',run_nb,k))
%     end
% end
% 
% %% Exporting images
% 
% files = dir('*.mat');
% for file = files'
%     if (not findstr(file.name,"array_radiograph") == [])
%         parsed = strsplit(file.name,"_");
%         run_nb = parsed(2)
%         k = parsed(3)
%         load(sprintf('%s_%d_%d.mat','array_radiograph',run_nb,k))
%         name_radiograph = sprintf('%s_%d_%d.png','radiograph',run_nb,k);
%         h_radiograph = figure;
%         set(gcf, 'Visible', 'off')
%         imagesc(edges_x/1e4,edges_x/1e4,S2/maximum_radio); colormap(flipud(colormap('gray')));
%         caxis([0, 1])
%         axis equal;
% end
% 
% 

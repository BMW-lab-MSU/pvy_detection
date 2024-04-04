clc; clear; close all;

model_files = dir('cost*.mat');
data_files = dir('compressed_virus_yes_no_img_*.mat');

save_dir = 'saved_predictions';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

parfor i = 1 : numel(model_files)
    model = load(fullfile(model_files(i).folder, model_files(i).name));
    fieldname = fieldnames(model);
    model = model.(fieldname{1});

    [~, model_folder, ~] = fileparts(model_files(i).name);
    if ~exist(fullfile(save_dir, model_folder), 'dir')
        mkdir(fullfile(save_dir, model_folder));
    end
    
    for j = 1 : numel(data_files)
        data_info = load(fullfile(data_files(j).folder, data_files(j).name));
        data = data_info.data_selected;
        labels = data_info.label_selected;
        idx = data_info.combined_indices + 1;
        pred = model.predictFcn(data);

        img_true = ones(1, 200 * 90) * 2;
        img_pred = img_true;

        img_true(idx) = labels;
        img_true = reshape(img_true, 90, 200)';

        img_pred(idx) = pred;
        img_pred = reshape(img_pred, 90, 200)';

        figure();
        subplot(121); imagesc(img_true); title('True Labels');
        c1 = colorbar('Location', 'southoutside'); clim([0 2]);
        c1.Ticks = [0, 1, 2]; c1.TickLabels = {'Healthy', 'Infected', 'Background'};

        subplot(122); imagesc(img_pred); title('Predicted Labels');
        c2 = colorbar('Location', 'southoutside'); clim([0 2]);
        c2.Ticks = [0, 1, 2]; c2.TickLabels = {'Healthy', 'Infected', 'Background'};
        
        image_num = regexp(data_files(j).name, 'img_(\d+)', 'tokens');
        image_num = image_num{1}{1};
        filename = fullfile(save_dir, model_folder, 'img_' + string(image_num) + '.jpg');
        saveas(gcf, filename);
        close;
    end
end
clc; clear; close all;

model_files = dir('./models_matlab/cost*.mat');
% data_files = dir('./potato_masks/*_10_downsampled_data.mat');
data_files = dir('./potato_masks/*_3_downsampled_data.mat');
down_dim_size = 3;

save_dir = 'predictions_matlab';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

parfor i = 1 : numel(model_files)
    model = load(fullfile(model_files(i).folder, model_files(i).name));
    fieldname = fieldnames(model);
    fprintf('Predicting with %s\n', fieldname{1});
    if strcmp(fieldname{1}, 'cost_1000_cubic_knn')
        continue
    end
    model = model.(fieldname{1});

    [~, model_folder, ~] = fileparts(model_files(i).name);
    if ~exist(fullfile(save_dir, model_folder), 'dir')
        mkdir(fullfile(save_dir, model_folder));
    end
    
    for j = 1 : numel(data_files)
        data_info = load(fullfile(data_files(j).folder, data_files(j).name));
        data = data_info.data;
        [~, data_name, ~] = fileparts(data_files(j).name);
        fprintf('Processing Image: %s\n', data_name);

        idx_info = load(fullfile(data_files(j).folder, strcat(data_name(1:end-5) , '_idx.mat')));
        idx = idx_info.idx + 1;

        height = floor(2000 / down_dim_size);
        width = floor(900 / down_dim_size);

        red = data(:, 113);
        green = data(:, 70);
        blue = data(:, 27);
        rgb_img = [red, green, blue];
        rgb_img = reshape(rgb_img, width, height, 3);
        rgb_img = rgb_img./ max(rgb_img);
        rgb_img = permute(rgb_img, [2, 1, 3]);

        data = data(idx, :);
        pred = model.predictFcn(data);
        num_infected = sum(pred);
        
        img_pred = ones(1, height * width)* 2;
        img_pred(idx) = pred;
        img_pred = reshape(img_pred, width, height)';

        figure();
        subplot(121);
        imagesc(rgb_img);
        axis image;
        axis off;
        subplot(122);
        imagesc(img_pred);
        axis image;
        axis off;

        img_name = fullfile(save_dir, model_folder, strcat(data_name, '_infected_' + string(num_infected) + '_in_' + string(length(idx)) + '.jpg'));        
        
        % saveas(gcf, img_name);
        print(gcf, img_name, '-djpeg', '-r300');
        close;
    end
end

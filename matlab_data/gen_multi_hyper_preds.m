clc; clear; close all;

%% format hyper data
data = readmatrix('compressed_virus_data.csv');
img_nums = unique(data(:, 1));
hyper_data = cell(1, numel(img_nums));

for i = 1 : numel(img_nums)
    cur_idx = data(:, 1) == img_nums(i);
    hyper_data{1, i} = data(cur_idx, 2 : end);
end

%% pred hyper data
models = {'hyper_ensemble_subspace_discriminant_89_9_74_1.mat', ...
    'hyper_linear_svm_89_5_76_2.mat'};

% models = {'multi_bilayered_NN_89_2_43_7.mat', ...
%     'multi_linear_svm_77_7_62_0.mat'};

save_dir = 'saved_predictions';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

for i = 1 : length(models)
    model = load(models{i});
    fieldname = fieldnames(model);
    model = model.(fieldname{1});

    [~, model_folder, ~] = fileparts(models{i});
    if ~exist(fullfile(save_dir, model_folder), 'dir')
        mkdir(fullfile(save_dir, model_folder));
    end
    
    for j = 1 : numel(hyper_data)
        idx = hyper_data{1, j}(:, 1);
        labels = hyper_data{1, j}(:, 2);
        pred = model.predictFcn(hyper_data{1, j}(:, 3 : end));

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
        
        image_num = img_nums(j);
        filename = fullfile(save_dir, model_folder, 'img_' + string(image_num) + '.jpg');
        saveas(gcf, filename);
        close;   
    end

end


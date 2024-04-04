clc; clear; close all;

data = readmatrix('compressed_virus_data.csv');
model = load('optimized_svm_susceptible_infected.mat');
fieldname = fieldnames(model);
model = model.(fieldname{1});
model_folder = fieldname{1};

save_dir = 'saved_predictions';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

if ~exist(fullfile(save_dir, model_folder), 'dir')
    mkdir(fullfile(save_dir, model_folder));
end

img_nums = unique(data(:, 1));

for i = 1 : length(img_nums)
    cur_image_num = img_nums(i);
    fprintf('Working on Image %d\n', cur_image_num);
    idx = find(data(:, 1) == cur_image_num);
    tmp_pixels = data(idx, 2);
    tmp_labels = data(idx, 3);
    tmp_data = data(idx, 4 : end);
    pred = model.predictFcn(tmp_data);
    acc = sum(tmp_labels == pred) / length(tmp_labels);

    img_true = ones(1, 200 * 90) * 2;
    img_pred = img_true;

    img_true(tmp_pixels) = tmp_labels;
    img_true = reshape(img_true, 90, 200)';

    img_pred(tmp_pixels) = pred;
    img_pred = reshape(img_pred, 90, 200)';

    figure();
    subplot(121); imagesc(img_true); title('True Labels');
    c1 = colorbar('Location', 'southoutside'); clim([0 2]);
    c1.Ticks = [0, 1, 2]; c1.TickLabels = {'Healthy', 'Infected', 'Background'};

    subplot(122); imagesc(img_pred); title('Predicted Labels');
    c2 = colorbar('Location', 'southoutside'); clim([0 2]);
    c2.Ticks = [0, 1, 2]; c2.TickLabels = {'Healthy', 'Infected', 'Background'};
    
    sgtitle("Image " + cur_image_num + ", Accuracy " + acc);
    filename = fullfile(save_dir, model_folder, 'img_' + string(cur_image_num) + '.jpg');
    saveas(gcf, filename);
    close;
end
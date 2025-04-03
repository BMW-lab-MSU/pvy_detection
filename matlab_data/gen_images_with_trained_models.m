clc; clear; close all;

data_files = dir('compressed_virus_yes_no_img_*.mat');
load('trained_models.mat');
model = results{6};     % this is the CNN model
save_dir = 'results_2024';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

modelTypes = {'SVM', 'DecisionTree', 'KNN', 'LogisticRegression', 'NeuralNetwork', 'CNN'};

model_folder = modelTypes{6};

if ~exist(fullfile(save_dir, model_folder), 'dir')
    mkdir(fullfile(save_dir, model_folder));
end


for j = 1 : numel(data_files)
    img_num = regexp(data_files(j).name, 'img_(\d+)_count', 'tokens');
    img_num = str2double(img_num{1}{1});
    fprintf('Working on Image %d\n', img_num);
    data_info = load(fullfile(data_files(j).folder, data_files(j).name));
    data = data_info.data_selected;
    feat = round(data(:, 1: 223));
    data = [data(:, 113), data(:, 70), data(:, 26)];
    
    labels = categorical(data_info.label_selected)';
    idx = data_info.combined_indices + 1;

    reshapedfeat = reshape(feat', [size(feat, 2), 1, 1, size(feat, 1)]);
    pred = classify(model, reshapedfeat);

    acc = sum(pred == labels) / numel(labels);

    % Initialize the output image with zeros
    img = zeros(200, 90, 3);
    
    % Loop through each channel
    for channel = 1:3
        % Create a temporary matrix with zeros for the current channel
        temp_channel = zeros(200 * 90, 1);
        
        % Place the values of data(:, channel) into the indices specified by idx
        temp_channel(idx) = data(:, channel);
        
        % Reshape the temporary matrix to the desired 200 x 90 shape
        img(:,:,channel) = reshape(temp_channel, 90, 200)';
    end

    % Find the minimum and maximum values of img
    min_val = min(img(:));
    max_val = max(img(:));
    
    % Normalize img to the range [0, 255]
    img = uint8((img - min_val) / (max_val - min_val) * 255);

    labels = double(labels) - 1;
    pred = double(pred) - 1;
    img_true = ones(1, 200 * 90) * 2;
    img_pred = img_true;

    img_true(idx) = labels;
    img_true = reshape(img_true, 90, 200)';

    img_pred(idx) = pred;
    img_pred = reshape(img_pred, 90, 200)';

    figure();
    subplot(131);
    imagesc(img); axis image; axis off; title('RGB Vegetation');
    % % Set custom colormap from black to green
    % colormap([linspace(0, 0, 256)', linspace(0, 1, 256)', linspace(0, 0, 256)']);
    % c3 = colorbar('Location', 'southoutside'); clim([0 255]);
    % c3.Ticks = [0, 255]; c3.TickLabels = {'Soil', 'Vegetation'};
    
    subplot(132); imagesc(img_true); title('True Labels');
    c1 = colorbar('Location', 'southoutside'); clim([0 2]);
    c1.Ticks = [0, 1, 2]; c1.TickLabels = {'Healthy', 'Infected', 'Background'};

    subplot(133); imagesc(img_pred); title('Predicted Labels');
    c2 = colorbar('Location', 'southoutside'); clim([0 2]);
    c2.Ticks = [0, 1, 2]; c2.TickLabels = {'Healthy', 'Infected', 'Background'};
    
    sgtitle("Image " + img_num + ", Accuracy " + acc);
    filename = fullfile(save_dir, model_folder, 'img_' + string(img_num) + '.jpg');
    saveas(gcf, filename);
    close;
end


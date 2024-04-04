clc; clear; close all;

data_files = dir('compressed_virus_yes_no_img_*.mat');
file_name = 'compressed_virus_data.csv';

for j = 1 : numel(data_files)
    img_num = regexp(data_files(j).name, 'img_(\d+)_count', 'tokens');
    img_num = str2double(img_num{1}{1});
    
    data_info = load(fullfile(data_files(j).folder, data_files(j).name));
    data = data_info.data_selected;
    labels = data_info.label_selected;
    idx = data_info.combined_indices + 1;

    img_true = ones(1, 90 * 200) * 2;

    img_true(idx) = labels;
    img_true = reshape(img_true, 90, 200);

    num_polygons = input(['How many polygons in ', num2str(img_num), ': ']);
    
    figure();
    imagesc(img_true);
    mask = zeros(size(img_true));

    for i = 1 : num_polygons
        h = drawpolygon('FaceAlpha', 0);
        tmp_mask = createMask(h);
        mask = mask | tmp_mask;
    end
    close();

    [roi_pixels, roi_idx] = intersect(idx, find(mask));
    roi_labels = labels(roi_idx);
    roi_data = data(roi_idx, :);
    img_num = repelem(img_num, length(roi_idx));

    csv_data = [img_num', roi_pixels, roi_labels', roi_data(:, 1 : 223)];

    if j == 1
        writematrix(csv_data, file_name);
    else
        writematrix(csv_data, file_name, 'WriteMode', 'append');
    end
end
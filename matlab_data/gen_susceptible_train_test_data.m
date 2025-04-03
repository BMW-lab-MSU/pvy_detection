clc; clear; close all;

data = readmatrix('compressed_virus_data.csv');

img_nums = unique(data(:, 1));

%% find the number of infected pixels per image

% for i = 1 : length(img_nums)
%     curr_img = img_nums(i);
%     curr_img_idx = find(data(:, 1) == curr_img);
%     num_infected = length(find(data(curr_img_idx, 3) == 1));
%     fprintf('Image %d, Infected %d out of %d\n', curr_img, ...
%         num_infected, length(curr_img_idx));
% end

% Image 21, Infected 1 out of 949
% Image 22, Infected 1118 out of 1341
% Image 25, Infected 111 out of 2314
% Image 26, Infected 54 out of 345      --> test
% Image 27, Infected 0 out of 649
% Image 29, Infected 0 out of 710
% Image 30, Infected 32 out of 1091
% Image 31, Infected 57 out of 831      --> test
% Image 32, Infected 0 out of 833
% Image 34, Infected 42 out of 1929
% Image 35, Infected 26 out of 2110
% Image 38, Infected 0 out of 3404
% Image 39, Infected 66 out of 1486     --> test
% Image 40, Infected 33 out of 115
% Image 42, Infected 0 out of 228
% Image 43, Infected 0 out of 1314      --> test
% Image 44, Infected 151 out of 1925
% Image 47, Infected 69 out of 69
% Image 48, Infected 34 out of 2201


%%
test_imgs = [26, 31, 39, 43];
train_imgs = setdiff(img_nums, test_imgs);

train_idx = find(ismember(data(:, 1), train_imgs));
test_idx = find(ismember(data(:, 1), test_imgs));

train_data = data(train_idx, 3 : end);
test_data = data(test_idx, 3 : end);



idx_1 = find(train_data(:, 1) == 1);
idx_0 = 1 : length(train_data(:, 1));
idx_0(idx_1) = [];

rng(0);
idx = randi(length(idx_0), 1, length(idx_1));

idx_0 = idx_0(idx);

data_0 = train_data(idx_0, :);
data_1 = train_data(idx_1, :);

train_data = [data_0; data_1];

rng(0);
idx = randperm(size(train_data, 1));
train_data = train_data(idx, :);

save('susceptible_train_test_data.mat', "train_data", "test_data");

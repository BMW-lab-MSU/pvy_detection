clc; clear; close all;

data = readmatrix('compressed_virus_data.csv');

train_data = data(1 : 18335, 3 : end);
test_data = data(18336 : end, 3 : end);

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


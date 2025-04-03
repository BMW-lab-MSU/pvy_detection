clc; clear; close all;

%% ready the hyper data
data = readmatrix('compressed_virus_data.csv');
hyper_train_idx = data(:, 1) == 21 | data(:, 1) == 22 | data(:, 1) == 26 | data(:, 1) == 31 | data(:, 1) == 34 | data(:, 1) == 39 | data(:, 1) == 40 | data(:, 1) == 42 | data(:, 1) == 43 | data(:, 1) == 44 | data(:, 1) == 47 | data(:, 1) == 48;
hyper_test_idx = ~(logical(1 : numel(data(:, 1)))' & hyper_train_idx);
hyper_trainX = data(hyper_train_idx, 4 : end);
hyper_trainY = data(hyper_train_idx, 3);
hyper_train = [data(hyper_train_idx, 4 : end), data(hyper_train_idx, 3)];

hyper_train0 = find(hyper_train(:, end) == 0);
hyper_train1 = find(hyper_train(:, end) == 1);

rng(0);
hyper_train0 = hyper_train0(randperm(length(hyper_train0)));
hyper_train = [hyper_train(hyper_train0(1 : numel(hyper_train1)), :); hyper_train(hyper_train1, :)];

rng(0);
hyper_train = hyper_train(randperm(size(hyper_train, 1)), :);

hyper_test = [data(hyper_test_idx, 4 : end), data(hyper_test_idx, 3)];

%% ready the multi data
% load('../mapir_multi_data/downsampled_multi.mat');
load('../mapir_multi_data/norm_downsampled_multi.mat');

% mat_train = [[trainX; valX,], [trainY, valY]'];
% mat_test = [testX, testY'];

subTrainX = trainX(11112 : end, :);
subTrainY = trainY(11112 : end);

multi_test = [trainX(1 : 11111, :), trainY(1 : 11111)'];

newTestImg = trainImg(1 : 11111);
multi_train = [[testX; subTrainX; valX], [testY, subTrainY, valY]'];

multi_train0 = find(multi_train(:, end) == 0);
multi_train1 = find(multi_train(:, end) == 1);

rng(0);
multi_train0 = multi_train0(randperm(length(multi_train0)));

multi_train = [multi_train(multi_train0(1 : numel(multi_train1)), :); multi_train(multi_train1, :)];

rng(0);
multi_train = multi_train(randperm(size(multi_train, 1)), :);




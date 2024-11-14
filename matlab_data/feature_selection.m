% clc; clear; close all;

data = readmatrix('compressed_virus_data.csv');
wv = readmatrix('compressed_virus_data_wv.csv');

% train_data = data(1 : 18335, 3 : end);
% test_data = data(18336 : end, 3 : end);
% 
% trainX = train_data(:, 2 : end);
% trainY = train_data(:, 1);
% testX = test_data(:, 2 : end);
% testY = test_data(:, 1);
% 
% train0 = find(trainY == 0);
% train1 = find(trainY == 1);
% 
% rng(0);
% train0 = train0(randperm(length(train0)));
% train1 = train1(randperm(length(train1)));

%% use different training set
hyper_train_idx = data(:, 1) == 21 | data(:, 1) == 22 | data(:, 1) == 26 | data(:, 1) == 31 | data(:, 1) == 34 | data(:, 1) == 39 | data(:, 1) == 40 | data(:, 1) == 42 | data(:, 1) == 43 | data(:, 1) == 44 | data(:, 1) == 47 | data(:, 1) == 48;
hyper_test_idx = ~(logical(1 : numel(data(:, 1)))' & hyper_train_idx);
hyper_trainX = data(hyper_train_idx, 4 : end);
hyper_trainY = data(hyper_train_idx, 3);
hyper_train = [data(hyper_train_idx, 4 : end), data(hyper_train_idx, 3)];

hyper_train0 = find(hyper_train(:, end) == 0);
hyper_train1 = find(hyper_train(:, end) == 1);
%%

num_samples = [100, 200, 400, 500, 700, 900, 1100, 1350, 1540];

count = 0;
for num = num_samples
    count = count + 1;
    % trainidx = [train0(1 : num); train1(1 : num)];
    
    trainidx = [hyper_train0(1 : num); hyper_train1(1 : num)];
    
    trainidx = trainidx(randperm(length(trainidx)));
    
    % subTrainX = trainX(trainidx, :);
    % subTrainY = trainY(trainidx);

    subTrainX = hyper_trainX(trainidx, :);
    subTrainY = hyper_trainY(trainidx);
    
%%
    
    % univariate feature ranking using chi-square tests
    [idx_chi, score_chi] = fscchi2(subTrainX, subTrainY);
    % idx_inf_chi = find(isinf(score_chi));
    % bar(score_chi(idx_chi));
    
    % filter type feature selection algorithms
    
    % rank feature using minimum redundancy maximum relevance
    [idx_mrmr, score_mrmr] = fscmrmr(subTrainX, subTrainY);
    
    % univariate feature ranking using F-test
    [idx_ftest, score_ftest] = fsrftest(subTrainX, subTrainY);
    
    % use rank faetures
    [idx_rank, score_rank] = rankfeatures(subTrainX', subTrainY);
    
    % feature selection using neighborhood component analysis
    mdl_nca = fscnca(subTrainX, subTrainY, 'Solver', 'sgd', 'Verbose', 1);
    % plot(mdl.FeatureWeights,'ro');
    [~, idx_nca] = sort(mdl_nca.FeatureWeights, 'descend');
    
    % rank imporance using ReliefF algorithm
    [idx_relieff, score_relieff] = relieff(subTrainX, subTrainY, 10, 'method', 'classification');
    
    score(1, :) = score_chi / max(score_chi);
    score(2, :) = score_mrmr / max(score_mrmr);
    score(3, :) = score_ftest / max(score_ftest);
    score(4, :) = score_rank / max(score_rank);
    score(5, :) = mdl_nca.FeatureWeights / max(mdl_nca.FeatureWeights);
    score(6, :) = score_relieff / max(score_relieff);
    
    idx_selections = 1 : 20;
    idx(1, :) = idx_chi(idx_selections);
    idx(2, :) = idx_mrmr(idx_selections);
    idx(3, :) = idx_ftest(idx_selections);
    idx(4, :) = idx_rank(idx_selections);
    idx(5, :) = idx_nca(idx_selections);
    idx(6, :) = idx_relieff(idx_selections);
    
    importance = zeros(size(idx, 1), size(subTrainX, 2));
    for i = 1 : size(idx, 1)
        importance(i, idx(i, :)) = score(i, (idx(i, :)));
    end
    
    cum_importance = sum(importance);
    cum_importance = cum_importance / max(cum_importance);
    
    subplot(3, 3, count);
    bar(wv, cum_importance);
    xlabel('Wavelength (nm)');
    ylabel('Normalized Predictor Importance');
    title(['Feature Importance for subset size ' num2str(num)]);
    
    % filename = ['feature_selection_plots/subset_size_' num2str(num) '_per_class.png'];
    % saveas(gcf, filename);
    % close();
end

%%

% Lambda = logspace(-7,-2,11);    % create 11 log-spaced regularization strenghts from 10^-7 to 10^-2.
% t = templateLinear('Learner', 'logistic', 'Solver', 'sgd', ...
%     'Regularization', 'lasso', 'GradientTolerance', 1e-8);
% 
% mdl = fitcecoc(subTrainX, subTrainY, 'Learners', t);

% beta parameter gives the weight of the features
% a = normalize(a, 'range', [0, 1]);    % maps to 0 and 1

%%
% mdl = fitcsvm(subTrainX, subTrainY);
% t = templateSVM("KernelFunction","gaussian","Type","classification");
% t = templateLinear('Learner', 'svm', 'Solver', 'sgd', ...
%     'Regularization', 'ridge', 'GradientTolerance', 1e-8);
% mdl = fitcecoc(subTrainX, subTrainY, 'Learners', t);



%%
% wrapper type feature selection feature algorithm

% sequential feature selection using custom criterion

% myfun = @(XTrain, YTrain, XTest, YTest) ...
%     size(XTest, 1) * loss(fitcecoc(XTrain, YTrain), XTest, YTest);
% 
% cv = cvpartition(subTrainY, 'KFold', 5);
% opts = statset('Display', 'iter', UseParallel=true);
% [tf_seq_ecoc, history_seq_ecoc] = sequentialfs(myfun, subTrainX, subTrainY, 'cv', cv, 'options', opts);
% 
% [tf_seq_corr, history_seq_corr] = sequentialfs(@mycorr, subTrainX, 'cv', 'none', 'options', opts);




% function criterion = mycorr(X)
%     p = size(X, 2);
%     R = corr(X, 'rows', 'pairwise');
%     R(logical(eye(p))) = NaN;
%     criterion = max(abs(R), [], 'all');
% end


%%
% Select Features for Classifying High-Dimensional Data
% Selecting Features Using a Simple Filter Approach
% kFoldCV = cvpartition(size(train_data, 1), "HoldOut", 1000);
% 
% dataTrainG0 = train_data(train_data(:, 1) == 0,:);
% dataTrainG1 = train_data(train_data(:, 1) == 1,:);
% [h,p,ci,stat] = ttest2(dataTrainG0,dataTrainG1,'Vartype','unequal');
% 
% [~,featureIdxSortbyP] = sort(p,2); % sort the features
% 
% classf = @(xtrain,ytrain,xtest,ytest) ...
%              sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
% 
% testMCE = zeros(1, 20);
% 
% for i = 1 : 20
%     fs = featureIdxSortbyP(1 : i);
%     testMCE(i) = crossval(classf, train_data(:, fs), train_data(:, 1), 'partition', kFoldCV);
% end
% 
% plot(1 : 20, testMCE, 'r^');
% xlabel('Number of Features');
% ylabel('MCE');




% function criterion = mycorr(X)
%     p = size(X, 2);
%     R = corr(X, 'rows', 'pairwise');
%     R(logical(eye(p))) = NaN;
%     criterion = max(abs(R), [], 'all');
% end
clc; clear; close all;

%{
    CHANGE THE VALUE OF THE taskID BELOW AS DESIRED
    
    taskID = 1 : Train with Normalized, Susceptible data
    taskID = 2 : Train Without Normalized, Susceptible data
%}

taskID = 1;    % UPDATE THE VALUE AS DESIRED



% Load the testing data
% Ensure this file/s is in the working directory
if taskID == 1
    disp('Training with Normalized, Susceptible data');
    load('susceptible_train_test_data.mat');
elseif taskID == 2
    disp('Training Without Normalized, Susceptible data');
    load('susceptible_no_norm_train_test_data.mat');
else
    error('Invalid Pointer!');
end

% Extract labels and features from the loaded data
data = double(train_data);
labels = categorical(data(:, 1)); % Convert class labels to categorical for deep learning
features = data(:, 2:end);        % Extract feature values (remaining columns)

% Verify that labels and features have compatible sizes
if size(features, 1) ~= numel(labels)
    error('Number of rows in features and number of labels must match.');
end

% Start a parallel pool if not already open
if isempty(gcp('nocreate'))
    parpool('local'); % Use 'local' profile
end

% Set up cross-validation partition for traditional ML models
cv = cvpartition(labels, 'KFold', 5);

% Define hyperparameter optimization settings for traditional models
hyperparams = struct('Optimizer', 'bayesopt', ...
                     'ShowPlots', false, ...
                     'CVPartition', cv, ...
                     'MaxObjectiveEvaluations', 30, ...
                     'AcquisitionFunctionName', 'expected-improvement-plus', ...
                     'UseParallel', false); % Disable parallel processing within bayesopt

% Define model types and initialize results cell array
modelTypes = {'SVM', 'DecisionTree', 'KNN', 'LogisticRegression', 'NeuralNetwork', 'CNN'};
results = cell(numel(modelTypes), 1);

runtimes = zeros(numel(modelTypes), 1);

% Train models in parallel
parfor i = 1:numel(modelTypes)
    modelType = modelTypes{i};
    fprintf('Training %s model...\n', modelType);
    
    % Initialize model variable to avoid uninitialized temporary warnings
    model = [];

    tStart = tic; % Start timer
    
    switch modelType
        case 'SVM'
            % Train an SVM with hyperparameter optimization
            model = fitcsvm(features, labels, ...
                            'Standardize', true, ...
                            'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale', 'KernelFunction'}, ...
                            'HyperparameterOptimizationOptions', hyperparams);
        
        case 'DecisionTree'
            % Train a Decision Tree with hyperparameter optimization
            model = fitctree(features, labels, ...
                             'OptimizeHyperparameters', {'MaxNumSplits', 'MinLeafSize', 'SplitCriterion'}, ...
                             'HyperparameterOptimizationOptions', hyperparams);
                             
        case 'KNN'
            % Train a k-NN model with hyperparameter optimization
            model = fitcknn(features, labels, ...
                            'OptimizeHyperparameters', {'NumNeighbors', 'Distance'}, ...
                            'HyperparameterOptimizationOptions', hyperparams);

        case 'LogisticRegression'
            % Train a logistic regression model with hyperparameter optimization
            model = fitclinear(features, labels, ...
                               'Learner', 'logistic', ...
                               'OptimizeHyperparameters', {'Lambda', 'Regularization'}, ...
                               'HyperparameterOptimizationOptions', hyperparams);

        case 'NeuralNetwork'
            % Define and train a feedforward neural network
            layers = [
                featureInputLayer(size(features, 2))
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(numel(categories(labels)))
                softmaxLayer
                classificationLayer];
            
            % Set training options
            options = trainingOptions('adam', ...
                                      'MaxEpochs', 50, ...
                                      'MiniBatchSize', 32, ...
                                      'Plots', 'none', ...
                                      'Shuffle', 'every-epoch');
            
            % Train the neural network
            model = trainNetwork(features, labels, layers, options);

        case 'CNN'
            % Reshape features for CNN input (1D CNN expects [Height, Width, Channels, numSamples])
            reshapedFeatures = reshape(features', [size(features, 2), 1, 1, size(features, 1)]);
            
            % Define a 1D convolutional neural network architecture
            layers = [
                imageInputLayer([size(features, 2), 1, 1])
                convolution2dLayer([3, 1], 16, 'Padding', 'same')
                batchNormalizationLayer
                reluLayer
                maxPooling2dLayer([2, 1], 'Stride', 2)
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(numel(categories(labels)))
                softmaxLayer
                classificationLayer];
            
            % Set training options
            options = trainingOptions('adam', ...
                                      'MaxEpochs', 50, ...
                                      'MiniBatchSize', 32, ...
                                      'Plots', 'none', ...
                                      'Shuffle', 'every-epoch', ...
                                      'ExecutionEnvironment', 'auto'); % Use GPU if available
            
            % Train the CNN
            model = trainNetwork(reshapedFeatures, labels, layers, options);
    end

    elapsedTime = toc(tStart); % End timer
    runtimes(i) = elapsedTime; % Store runtime
    
    % Store the model in results cell array
    results{i} = model;
end

% save all the trained models together
if taskID == 1
    save('trained_models.mat', "results");
    save('model_runtimes.mat', 'modelTypes', 'runtimes');
elseif taskID == 2
    save('trained_models_no_norm.mat', "results");
    save('model_runtimes_no_norm.mat', 'modelTypes', 'runtimes');
end

% Save each model outside the parfor loop
for i = 1:numel(modelTypes)
    modelType = modelTypes{i};
    model = results{i}; % Retrieve each model from results cell
    if taskID == 1
        save(sprintf('%s_model_results.mat', modelType), 'model');
    elseif taskID == 2
        save(sprintf('%s_model_results_no_norm.mat', modelType), 'model');
    end
    fprintf('%s model training completed and saved.\n', modelType);
end

disp('All models have been trained and saved.');

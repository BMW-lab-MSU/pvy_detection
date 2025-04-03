clc; clear; close all;

disp('Loading Data')
load('susceptible_train_test_data.mat');

% Load the data
data = train_data;          % Replace 'data.csv' with your file name
% labels = data(:, 1);        % Extract the class labels (first column)
labels = categorical(data(:, 1)); % Convert class labels to categorical for deep learning
features = data(:, 2:end);  % Extract the feature values (remaining columns)

% Reshape features for CNN if needed (e.g., treating 223 values as 1D "image" with 1 channel)
inputSize = [223, 1, 1]; % Adjust dimensions if you need a different input format

% Start a parallel pool (adjust the number of workers if needed)
if isempty(gcp('nocreate'))
    parpool('local'); % Use 'local' profile
end

% Set up cross-validation partition for traditional ML models
cv = cvpartition(data(:, 1), 'KFold', 5);

% Define hyperparameter optimization settings for traditional models
hyperparams = struct('Optimizer', 'bayesopt', ...
                     'ShowPlots', false, ...
                     'CVPartition', cv, ...
                     'MaxObjectiveEvaluations', 30, ...
                     'AcquisitionFunctionName', 'expected-improvement-plus', ...
                     'UseParallel', false); % Enable parallel processing within models

% Define model types and their respective options
modelTypes = {'SVM', 'DecisionTree', 'KNN', 'LogisticRegression', 'NeuralNetwork', 'CNN'};
results = cell(numel(modelTypes), 1);

% Train models in parallel
parfor i = 1:numel(modelTypes)
    modelType = modelTypes{i};
    fprintf('Training %s model...\n', modelType);

    % Initialize model variable to avoid uninitialized temporary warnings
    model = [];
    
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
                featureInputLayer(223)
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(2)
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
            % Reshape features for CNN if required (223x1x1, treating it as a "1D image")
            reshapedFeatures = reshape(features, [size(features, 1), inputSize]);
            
            % Define a 1D convolutional neural network architecture
            layers = [
                imageInputLayer(inputSize)
                convolution2dLayer([3, 1], 16, 'Padding', 'same')
                batchNormalizationLayer
                reluLayer
                maxPooling2dLayer([2, 1], 'Stride', 2)
                fullyConnectedLayer(64)
                reluLayer
                fullyConnectedLayer(2)
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
    
    % % Save the model and its results
    % Store the model in results cell array
    results{i} = model;
    % save(sprintf('%s_model_results.mat', modelType), 'model');
    % fprintf('%s model training completed and saved.\n', modelType);
end

% Save each model outside the parfor loop
for i = 1:numel(modelTypes)
    modelType = modelTypes{i};
    model = results{i}; % Retrieve each model from results cell
    save(sprintf('%s_model_results.mat', modelType), 'model');
    fprintf('%s model training completed and saved.\n', modelType);
end

disp('All models have been trained and saved.');





%%
% % Set up cross-validation options
% cv = cvpartition(labels, 'KFold', 5); % 5-fold cross-validation
% 
% % Define SVM template with optimization
% svmTemplate = templateSVM('KernelFunction', 'linear', 'Standardize', true);
% 
% % Define the hyperparameter optimization options
% hyperparams = struct('Optimizer', 'bayesopt', ... % Use Bayesian optimization
%                      'ShowPlots', true, ...
%                      'CVPartition', cv, ...
%                      'MaxObjectiveEvaluations', 30, ... % Number of evaluations
%                      'AcquisitionFunctionName', 'expected-improvement-plus', ...
%                      'UseParallel', true); % Enable parallel processing
% 
% disp('Starting model optimization')
% % Train SVM with optimization
% svmModel = fitcsvm(features, labels, ...
%                    'Standardize', true, ...
%                    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale', 'KernelFunction'}, ...
%                    'HyperparameterOptimizationOptions', hyperparams);
% 
% % % Display the results
% disp('Best SVM Model:')
% disp(svmModel)
% 
% % Save the optimized model and results to a .mat file
% % save('svm_model_results.mat', 'svmModel');
% disp('Optimization and training completed. Model saved to svm_model_results.mat');

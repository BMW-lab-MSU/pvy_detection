clc; clear; close all;

disp('Loading Data')

% Load the training and testing data
load('susceptible_train_test_data.mat'); % Ensure this file is in the working directory

% % Extract labels and features from the loaded data
% train_data = susceptible_train_test_data.train_data;
% test_data = susceptible_train_test_data.test_data;

% Convert labels to categorical
train_labels = categorical(train_data(:, 1)); % Labels for training
test_labels = categorical(test_data(:, 1));   % Labels for testing
train_features = train_data(:, 2:end);        % Features for training
test_features = test_data(:, 2:end);          % Features for testing

% Verify that train_features and test_features have matching sizes
if size(train_features, 1) ~= numel(train_labels)
    error('Number of rows in train_features and number of train_labels must match.');
end
if size(test_features, 1) ~= numel(test_labels)
    error('Number of rows in test_features and number of test_labels must match.');
end

% Load the pre-trained models from the results cell array
load('trained_models.mat', 'results'); % Assuming 'results' is a 6x1 cell containing trained models

% Model types
modelTypes = {'SVM', 'DecisionTree', 'KNN', 'LogisticRegression', 'NeuralNetwork', 'CNN'};

% Initialize a structure to store the results
modelResults = struct();

% Loop to generate predictions, calculate accuracies, and create confusion matrices
for i = 1:numel(modelTypes)
    modelType = modelTypes{i};
    model = results{i}; % Load the model from the cell array
    
    fprintf('Using %s model...\n', modelType);
    
    % Predict on the test data based on the model type
    switch modelType
        case {'SVM', 'DecisionTree', 'KNN', 'LogisticRegression'}
            % For traditional models, use the predict function
            testPred = predict(model, test_features);
        case {'NeuralNetwork', 'CNN'}
            % For Neural Network and CNN, reshape features and classify
            if strcmp(modelType, 'NeuralNetwork')
                testPred = classify(model, test_features); % Feedforward NN
            else
                % Reshape the features for CNN input (assuming you have used a similar structure before)
                reshapedTestFeatures = reshape(test_features', [size(test_features, 2), 1, 1, size(test_features, 1)]);
                testPred = classify(model, reshapedTestFeatures);
            end
        otherwise
            error('Unknown model type: %s', modelType);
    end
    
    % Calculate accuracy
    testAcc = sum(testPred == test_labels) / numel(test_labels);
    
    % Generate confusion matrix
    confusionMatrix = confusionmat(test_labels, testPred);
    
    % Extract TP, FP, FN, TN from confusion matrix
    TP = confusionMatrix(1, 1); % True Positives
    FP = confusionMatrix(1, 2); % False Positives
    FN = confusionMatrix(2, 1); % False Negatives
    TN = confusionMatrix(2, 2); % True Negatives
    
    % Calculate Precision and Recall
    Precision = TP / (TP + FP);
    Recall = TP / (TP + FN);
    
    % Calculate F1 Score
    F1_Score = 2 * (Precision * Recall) / (Precision + Recall);
    
    % Save results in the modelResults structure
    modelResults.(modelType).testPred = testPred;
    modelResults.(modelType).testAcc = testAcc;
    modelResults.(modelType).confusionMatrix = confusionMatrix;
    modelResults.(modelType).Precision = Precision;
    modelResults.(modelType).Recall = Recall;
    modelResults.(modelType).F1_Score = F1_Score; % Add F1 score
    
    % Save confusion matrix to a separate file
    % save(sprintf('%s_confusion_matrix.mat', modelType), 'confusionMatrix');
    
    % Print out the results
    fprintf('%s Model - Test Accuracy: %.4f\n', modelType, testAcc);
end

% Save all results to a file
save('updated_trained_models.mat', 'modelResults');

disp('Predictions, accuracies, and confusion matrices have been saved.');

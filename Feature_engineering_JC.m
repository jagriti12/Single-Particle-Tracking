%% Setup for reproducibility
rng(0);
tic

% Convert train and test data to table for feature selection
combined_train_ftrs_lbls = vertcat(train_ftrs, train_lbls);
combined_train_ftrs_lbls = transpose(combined_train_ftrs_lbls);
combined_test_ftrs_lbls = vertcat(test_ftrs, test_lbls);
combined_test_ftrs_lbls = transpose(combined_test_ftrs_lbls);

%% Define Variable Names for Features and Label
variableNames = {'Alpha', 'AngularGaussianityIndex', 'Asymmetry', 'AvgMSDRatio', 'Efficiency',...
                 'FractalDimension', 'Gaussianity', 'JumpLength', 'Kurtosis', 'MaximalExcursion',...
                 'MeanMaximalExcursion', 'Straightness', 'Trappedness', 'VelocityAutocorrelation','Labels'};
labelColumnName = 'Labels';

%% Convert Arrays to Tables with Appropriate Variable Names
T_features_train = array2table(combined_train_ftrs_lbls(:, 1:end-1), 'VariableNames', variableNames(1:end-1));
T_features_train.(labelColumnName) = combined_train_ftrs_lbls(:, end);

T_features_test = array2table(combined_test_ftrs_lbls(:, 1:end-1), 'VariableNames', variableNames(1:end-1));
T_features_test.(labelColumnName) = combined_test_ftrs_lbls(:, end);

% Initialize results storage
results = struct('Method', [], 'Accuracy', [], 'Model', [], 'Features', []);

% Create a waitbar
wb = waitbar(0, 'Initializing...');

%% MRMR Feature Selection
waitbar(0.1, wb, 'Performing MRMR feature selection...');
[idxMRMR, scoresMRMR] = fscmrmr(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName));
waitbar(0.2, wb, 'MRMR feature selection completed.');

% Select top 3 features
topFeaturesMRMR = idxMRMR(1:3);

%% NCA Feature Selection
waitbar(0.3, wb, 'Performing NCA feature selection...');
mdlNCA = fscnca(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName), 'Solver', 'lbfgs');
waitbar(0.4, wb, 'NCA feature selection completed.');

% Select top 3 features
[~, sortedIndices] = sort(mdlNCA.FeatureWeights, 'descend');
topFeaturesNCA = sortedIndices(1:3);

%% ReliefF Feature Selection
waitbar(0.5, wb, 'Performing ReliefF feature selection...');
[idxReliefF, weightsReliefF] = relieff(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName), 10);
waitbar(0.6, wb, 'ReliefF feature selection completed.');

% Select top 3 features
topFeaturesReliefF = idxReliefF(1:3);

%% Combine Top Features from MRMR, NCA, and ReliefF for the Combined Model
combinedTopFeatures = unique([topFeaturesMRMR(:); topFeaturesNCA(:); topFeaturesReliefF(:)]);

% Correct the histogram computation to use the correct table
featureCounts = histcounts(combinedTopFeatures, 1:numel(T_features_train.Properties.VariableNames)+1);

% Since we're now dealing with indices directly, adjust the plotting code accordingly
numBins = numel(T_features_train.Properties.VariableNames);
figure;
bar(1:numBins, featureCounts);
xlabel('Feature Index');
ylabel('Feature Occurrences');
title('Combined Feature Occurrences from Different Algorithms');
xticks(1:numBins);
xticklabels(T_features_train.Properties.VariableNames);
xtickangle(45);
ylim([0, max(featureCounts) + 1]);
grid on;

% Define training and test labels for convenience
y_train = T_features_train.(labelColumnName);
y_test = T_features_test.(labelColumnName);

% Initialize accuracy storage
accuracies = zeros(4, 1);


%% MRMR Features Model
waitbar(0.7, wb, 'Training and evaluating model with MRMR features...');
X_train_MRMR = table2array(T_features_train(:, variableNames(topFeaturesMRMR)));
X_test_MRMR = table2array(T_features_test(:, variableNames(topFeaturesMRMR)));
accuracies(1) = trainAndEvaluateSubspaceKNN(X_train_MRMR, y_train, X_test_MRMR, y_test);
results(1).Method = 'MRMR';
results(1).Accuracy = accuracies(1);
results(1).Features = variableNames(topFeaturesMRMR);
fprintf('Accuracy with MRMR Top Features (Subspace KNN): %.2f%%\n', accuracies(1) * 100);

%% NCA Features Model
waitbar(0.8, wb, 'Training and evaluating model with NCA features...');
X_train_NCA = table2array(T_features_train(:, variableNames(topFeaturesNCA)));
X_test_NCA = table2array(T_features_test(:, variableNames(topFeaturesNCA)));
accuracies(2) = trainAndEvaluateSubspaceKNN(X_train_NCA, y_train, X_test_NCA, y_test);
results(2).Method = 'NCA';
results(2).Accuracy = accuracies(2);
results(2).Features = variableNames(topFeaturesNCA);
fprintf('Accuracy with NCA Top Features (Subspace KNN): %.2f%%\n', accuracies(2) * 100);

%% ReliefF Features Model
waitbar(0.9, wb, 'Training and evaluating model with ReliefF features...');
X_train_ReliefF = table2array(T_features_train(:, variableNames(topFeaturesReliefF)));
X_test_ReliefF = table2array(T_features_test(:, variableNames(topFeaturesReliefF)));
accuracies(3) = trainAndEvaluateSubspaceKNN(X_train_ReliefF, y_train, X_test_ReliefF, y_test);
results(3).Method = 'ReliefF';
results(3).Accuracy = accuracies(3);
results(3).Features = variableNames(topFeaturesReliefF);
fprintf('Accuracy with ReliefF Top Features (Subspace KNN): %.2f%%\n', accuracies(3) * 100);

%% Combined Features Model
waitbar(1.0, wb, 'Training and evaluating model with combined features...');
combinedFeatures = variableNames(combinedTopFeatures); % Using unique combined feature indices
X_train_Combined = table2array(T_features_train(:, combinedFeatures));
X_test_Combined = table2array(T_features_test(:, combinedFeatures));
accuracies(4) = trainAndEvaluateSubspaceKNN(X_train_Combined, y_train, X_test_Combined, y_test);
results(4).Method = 'Combined';
results(4).Accuracy = accuracies(4);
results(4).Features = combinedFeatures;
fprintf('Accuracy with Combined Top Features (Subspace KNN): %.2f%%\n', accuracies(4) * 100);

% Save the results
save('feature_selection_results.mat', 'results');
toc
% Close the waitbar
close(wb);

% Function to train and evaluate a Subspace KNN model
function accuracy = trainAndEvaluateSubspaceKNN(X_train, y_train, X_test, y_test)
    mdl = fitcsubspace(X_train, y_train, 'Learners', 'knn');
    y_pred = predict(mdl, X_test);
    accuracy = sum(y_pred == y_test) / length(y_test);
end
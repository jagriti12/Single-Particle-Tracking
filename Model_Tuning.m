rng(0);
 %% MODEL TUNING ON THE COMBINED TOP FEATURES %%
X_train_Combined;
X_test_Combined;
y_train;
y_test;
Layers =5;
k = 5; % Number of folds

%% Single Tree with CV %%
cvSingleTree = fitctree(X_train_Combined, y_train, 'CrossVal', 'on', 'KFold', k);
cvSingleTreeLoss = kfoldLoss(cvSingleTree);
predictionsSingleTree = kfoldPredict(cvSingleTree);
accuracySingleTree = sum(predictionsSingleTree == y_train) / length(y_train);
%% Random Forest with Manual CV %%

cvpart = cvpartition(y_train, 'KFold', k);
oobErrorRF = zeros(k, 1);

for i = 1:k
    trainingIndices = training(cvpart, i);
    testIndices = test(cvpart, i);
    modelRF = TreeBagger(50, X_train_Combined(trainingIndices, :), y_train(trainingIndices, :), ...
                         'Method', 'classification', 'OOBPrediction', 'On', ...
                         'MaxNumSplits', 2^Layers - 1);
    oobErrorRF(i) = oobError(modelRF, 'Mode', 'ensemble');
end

meanOOBErrorRF = mean(oobErrorRF);
accuracyRF = 1 - meanOOBErrorRF;
%% Boosted Trees %%
% cvBoostedTrees = fitcensemble(X_train_Combined, y_train, 'Method', 'RUSBoost', ...
%                               'CrossVal', 'on', 'KFold', k);
% cvBoostedLoss = kfoldLoss(cvBoostedTrees);
% predictionsBoosted = kfoldPredict(cvBoostedTrees);
% accuracyBoosted = sum(predictionsBoosted == y_train) / length(y_train);

%% Bagged Trees %%
cvBaggedTrees = fitcensemble(X_train_Combined, y_train, 'Method', 'Bag', ...
                             'CrossVal', 'on', 'KFold', k);
cvBaggedLoss = kfoldLoss(cvBaggedTrees);
predictionsBagged = kfoldPredict(cvBaggedTrees);
accuracyBagged = sum(predictionsBagged == y_train) / length(y_train);



% Print out accuracies for comparison
% fprintf('Accuracy of Single Decision Tree: %.2f%%\n', accuracySingleTree * 100);
% fprintf('Accuracy of Random Forest: %.2f%%\n', accuracyRF * 100);
% fprintf('Accuracy of Boosted Trees: %.2f%%\n', accuracyBoosted * 100);
fprintf('Accuracy of Bagged Trees: %.2f%%\n', accuracyBagged * 100);
toc
% Print out losses for comparison
fprintf('Cross-Validated Loss of Single Decision Tree: %.2f%%\n', cvSingleTreeLoss * 100);
fprintf('Mean OOB Error of Random Forest: %.2f%%\n', meanOOBErrorRF * 100);
fprintf('Cross-Validated Loss of Boosted Trees: %.2f%%\n', cvBoostedLoss * 100);
fprintf('Cross-Validated Loss of Bagged Trees: %.2f%%\n', cvBaggedLoss * 100);

%% 

% Store the CV results and accuracies in a structured array
cvResults = struct('Model', {'Single Decision Tree', 'Random Forest', 'Boosted Trees', 'Bagged Trees'}, ...
                   'CVLoss', {cvSingleTreeLoss, meanOOBErrorRF, cvBoostedLoss, cvBaggedLoss}, ...
                   'Accuracy', {accuracySingleTree, accuracyRF, accuracyBoosted, accuracyBagged});

% Analyze the results to find the model with the lowest cross-validation loss
[~, bestLossIndex] = min([cvResults.CVLoss]);
bestModelByLoss = cvResults(bestLossIndex);

% Analyze the results to find the model with the highest accuracy
[~, bestAccuracyIndex] = max([cvResults.Accuracy]);
bestModelByAccuracy = cvResults(bestAccuracyIndex);

% Displaying the Best Models based on Loss and Accuracy
fprintf('The best model based on cross-validation loss is: %s with a loss of %.2f%%\n', ...
        bestModelByLoss.Model, bestModelByLoss.CVLoss * 100);
fprintf('The best model based on accuracy is: %s with an accuracy of %.2f%%\n', ...
        bestModelByAccuracy.Model, bestModelByAccuracy.Accuracy * 100);

% If the best model by loss is also the best model by accuracy, it's the overall best.
% Otherwise, additional considerations may be needed.
if bestLossIndex == bestAccuracyIndex
    fprintf('Overall Best Model considering both Loss and Accuracy: %s\n', bestModelByAccuracy.Model);
else
    fprintf('A decision needs to be made between Loss and Accuracy criteria as they suggest different best models.\n');
end
%% 

%Plotting%
% Create a figure for the plot
figure;

% Setting up x-axis and width for bars
x = 1:numel(cvResults);  % X locations of the bars
width = 0.3;

% Plotting CV Loss
yyaxis left;
for i = 1:numel(cvResults)
    if i == bestLossIndex
        bar(x(i) - width/2, cvResults(i).CVLoss, width, 'FaceColor', 'cyan'); % Highlight best model by CV Loss
    else
        bar(x(i) - width/2, cvResults(i).CVLoss, width, 'FaceColor', 'blue');
    end
    hold on; % Keep the plot for the next series of bars
end
ylabel('Cross-Validated Loss (%)');

% Plotting Accuracy
yyaxis right;
for i = 1:numel(cvResults)
    if i == bestAccuracyIndex
        bar(x(i) + width/2, cvResults(i).Accuracy, width, 'FaceColor', 'magenta'); % Highlight best model by Accuracy
    else
        bar(x(i) + width/2, cvResults(i).Accuracy, width, 'FaceColor', 'red');
    end
    hold on; % Keep the plot for the next series of bars
end
ylabel('Accuracy (%)');

% Finalizing the plot
hold off; % No more plots
set(gca, 'XTick', x, 'XTickLabel', {cvResults.Model}, 'XTickLabelRotation', 45);
xlabel('Model');
title('Model Performance: CV Loss vs Accuracy');
legend({'CV Loss', 'Accuracy'}, 'Location', 'Best');
grid on;
%% 

% Determine the best model based on the given criteria
bestModelIndex = bestAccuracyIndex; % Change this if you want to prioritize CV loss over accuracy

% Get the best model
bestModel = cvResults(bestModelIndex);

% Print the chosen best model
fprintf('The best model is: %s\n', bestModel.Model);
%% 
%% 

rng(0); % Fixing the random seed for reproducibility

% Train the best model on the entire training data (assuming Bagged Trees is the best model)

bestModelTrained = fitcensemble(X_train_Combined, y_train, 'Method', 'Bag');
[y_pred, scores] = predict(bestModelTrained, X_test_Combined);



% Initialize colors for plotting
colors = lines(numel(unique(y_test))); % This generates a colormap with a distinct color for each class

% Plot the Confusion Matrix
figure;
confMat = confusionmat(y_test, y_pred);
confusionchart(confMat);
title('Confusion Matrix');

% Number of classes
num_classes = numel(unique(y_test));

% Prepare to store ROC data and AUC values
legend_labels = cell(num_classes, 1);
TPR_all = cell(num_classes, 1);
FPR_all = cell(num_classes, 1);
AUC_values = zeros(num_classes, 1);

% Iterate over each class to compute TPR, FPR, and AUC
for i = 1:num_classes
    % Calculate scores for the current class
    scores_current = scores(:, i);
    true_labels = (y_test == i);

    % Sort the scores and true labels
    [~, idx] = sort(scores_current, 'descend');
    sorted_labels = true_labels(idx);

    % Calculate cumulative sums for true and false positives
    TP = cumsum(sorted_labels);
    FP = cumsum(~sorted_labels);

    % Normalize cumulative sums to get TPR and FPR
    TPR_comp = TP / sum(sorted_labels);
    FPR_comp = FP / sum(~sorted_labels);

    % Store TPR and FPR for later use in plotting
    TPR_all{i} = TPR_comp;
    FPR_all{i} = FPR_comp;

    % Calculate AUC using the trapezoidal rule
    AUC = trapz(FPR_comp, TPR_comp);
    AUC_values(i) = AUC;

    % Prepare legend label with AUC value
    legend_labels{i} = sprintf('Class %d (AUC = %.4f)', i, AUC);
end

% Plot the ROC curves
figure;
hold on;
for i = 1:num_classes
    plot(FPR_all{i}, TPR_all{i}, 'Color', colors(i, :), 'LineWidth', 2);
end
legend(legend_labels, 'Location', 'Best');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves by Class');
grid on;
hold off;

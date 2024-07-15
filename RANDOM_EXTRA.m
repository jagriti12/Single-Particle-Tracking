rng(0); % Set random number generator seed for reproducibility
tic;

% Assuming these variables are already defined
X_train_Combined;
X_test_Combined;
y_train;
y_test;

% Number of folds for cross-validation
k = 5; 
cvpart = cvpartition(y_train, 'KFold', k);

% Define the ranges for each hyperparameter
numLearningCyclesGrid = 50:500;
minLeafSizeGrid = 10:50;

% Number of random samples to draw
numSamples = 100; % for example, 100 iterations of random combinations

% Initialize waitbar
wb = waitbar(0, 'Initializing random search...');

% Prepare to store results
results = [];
iterationCount = 0;

% Random Search
for i = 1:numSamples
    % Randomly select parameters from the defined ranges
    numCycles = randsample(numLearningCyclesGrid, 1);
    leafSize = randsample(minLeafSizeGrid, 1);

    % Update waitbar
    waitbar(iterationCount / numSamples, wb, sprintf('Evaluating: Cycles=%d, LeafSize=%d', numCycles, leafSize));
    
    % Cross-validation setup
    cvLosses = zeros(k, 1);

    for j = 1:k
        trainIndices = training(cvpart, j);
        validationIndices = test(cvpart, j);
        
        % Train the model with the current set of parameters
        t = templateTree('MinLeafSize', leafSize);
        model = fitcensemble(X_train_Combined(trainIndices, :), y_train(trainIndices), ...
            'Method', 'Bag', 'NumLearningCycles', numCycles, 'Learners', t);
        
        % Evaluate the model on the validation set
        validationPredictions = predict(model, X_train_Combined(validationIndices, :));
        cvLosses(j) = sum(validationPredictions ~= y_train(validationIndices)) / length(validationIndices);
    end
    
    % Calculate average CV loss
    averageCvLoss = mean(cvLosses);
    results = [results; struct('NumLearningCycles', numCycles, 'MinLeafSize', leafSize, 'CvLoss', averageCvLoss)];
    
    % Update iteration count
    iterationCount = iterationCount + 1;
end

% Close waitbar
close(wb);

% Find the best parameters
[~, bestIdx] = min([results.CvLoss]);
bestParameters = results(bestIdx);

% Train the model with the best parameters
bestModel = fitcensemble(X_train_Combined, y_train, 'Method', 'Bag', ...
    'NumLearningCycles', bestParameters.NumLearningCycles, 'Learners', templateTree('MinLeafSize', bestParameters.MinLeafSize));

% Make predictions on the test set
predictions = predict(bestModel, X_test_Combined);
optimizedAccuracy = sum(predictions == y_test) / length(y_test);
fprintf('Optimized Accuracy: %.2f%%\n', optimizedAccuracy * 100);

% Plot confusion matrix
figure;
confMat = confusionchart(y_test, predictions);
confMat.Title = 'Confusion Matrix - Optimized Model';

toc;
%% 

[y_pred, scores] = predict(bestModel, X_test_Combined);



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
    TPR_random = TP / sum(sorted_labels);
    FPR_random = FP / sum(~sorted_labels);

    % Store TPR and FPR for later use in plotting
    TPR_all{i} = TPR_random;
    FPR_all{i} = FPR_random;

    % Calculate AUC using the trapezoidal rule
    AUC = trapz(FPR_random, TPR_random);
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

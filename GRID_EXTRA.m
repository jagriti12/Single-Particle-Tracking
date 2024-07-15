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
numLearningCyclesGrid = (50:500);
minLeafSizeGrid = (10:50);

% Initialize waitbar
totalIterations = length(numLearningCyclesGrid) * length(minLeafSizeGrid);
wb = waitbar(0, 'Initializing grid search...');

% Prepare to store results
results = [];
iterationCount = 0;

% Grid Search
for numCycles = numLearningCyclesGrid
    for leafSize = minLeafSizeGrid
        % Update waitbar
        waitbar(iterationCount / totalIterations, wb, sprintf('Evaluating: Cycles=%d, LeafSize=%d', numCycles, leafSize));
        
        % Cross-validation setup
        cvLosses = zeros(k, 1);

        for i = 1:k
            trainIndices = training(cvpart, i);
            validationIndices = test(cvpart, i);
            
            % Train the model with the current set of parameters
            t = templateTree('MinLeafSize', leafSize);
            model = fitcensemble(X_train_Combined(trainIndices, :), y_train(trainIndices), ...
                'Method', 'Bag', 'NumLearningCycles', numCycles, 'Learners', t);
            
            % Evaluate the model on the validation set
            validationPredictions = predict(model, X_train_Combined(validationIndices, :));
            cvLosses(i) = sum(validationPredictions ~= y_train(validationIndices)) / length(validationIndices);
        end
        
        % Calculate average CV loss
        averageCvLoss = mean(cvLosses);
        results = [results; struct('NumLearningCycles', numCycles, 'MinLeafSize', leafSize, 'CvLoss', averageCvLoss)];
        
        % Update iteration count
        iterationCount = iterationCount + 1;
    end
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

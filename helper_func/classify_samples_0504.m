
function predicted_classes = classify_samples_0504(test_sample_matrix, B)
    % classify_samples: Classify test samples using the learned B matrix
    %
    % Inputs:
    % - test_sample_matrix: Test samples matrix (n_test samples x k factors)
    % - B: Classification matrix (k factors x num_classes + 1) with constant term
    %
    % Outputs:
    % - predicted_classes: Predicted classes for the test samples (1 x n_test)

    % Calculate the linear and constant parts separately
    linear_part = test_sample_matrix * B(1:end-1, :);
    constant_part = repmat(B(end, :), size(test_sample_matrix, 1), 1);
    total_part = linear_part + constant_part;

    % Compute the softmax probabilities
    softmax_probabilities = exp(total_part) ./ sum(exp(total_part), 2);

    % Find the class with the highest probability for each sample
    [~, predicted_classes] = max(softmax_probabilities, [], 2);

    % Reshape the output to a row vector
    predicted_classes = predicted_classes';
end

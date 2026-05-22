function [W_best, H_best, B_best, pred_integrated, pred_two_step, loss_history, test_acc_history, best_acc] = ...
    supervised_NMF(...
    X_train, Y_train, X_test, Y_test, k, gamma, lambda_B, alpha, max_iter)


% Inputs:
% - X_train: Input matrix (n samples x m features)
% - Y_train: Class label matrix for each sample (1 x n)
% - X_test, Y_test: Test set and labels
% - k: Number of factors
% - gamma: Regularization parameter to balance reconstruction and classification loss
% - lambda_B: Regularization parameter for L2 regularization of B
% - max_iter: Maximum number of iterations
%
% Outputs:
% - W_best, H_best, B_best: The best W, H, B found
% - pred_integrated: The integrated predictions on the test set corresponding to B_best
% - pred_two_step: The predictions obtained by a two-step approach (logistic regression on W)
% - loss_history: The recorded history of loss values
% - test_acc_history: The recorded history of test accuracies
% - best_acc: The best test accuracy achieved

[n, m] = size(X_train);
W = abs(rand(n, k));
H = abs(rand(k, m));

% Normalize initial W and H
W = W ./ sum(W, 1);
H = H ./ sum(H, 2);

% Convert class labels to one-hot encoding
Y_one_hot = full(ind2vec(Y_train'));

% Initialize B with non-negative random values and an additional row for the constant term
num_classes = size(Y_one_hot, 1);
B = abs(rand(k + 1, num_classes));

% Simple Gradient Descent parameter for B


loss_history = nan(1, max_iter);
test_acc_history = nan(1, max_iter);
chk_period = 5;
lsq_options = optimset('Display','none');

for iter = 1:max_iter

    % Update H
    WH = W * H;
    H = H .* (W' * X_train) ./ (W' * WH);

    % Update W
    WB = [W, ones(n, 1)] * B;
    max_WB = max(WB, [], 2);
    W_softmax = exp(WB - max_WB) ./ sum(exp(WB - max_WB), 2);
    W = W .* ((X_train ./ WH) * H' + gamma * (Y_one_hot' - W_softmax) * B(1:end-1, :)') ...
        ./ (ones(n, m) * H');

    % Non-negativity constraints for W
    W = max(W, 0);


    % Clip W and H values
    W = max(min(W, 1e10), 1e-10);
    H = max(min(H, 1e10), 1e-10);

    % Update B using Simple Gradient Descent
    g_B = -([W, ones(n, 1)]' * (Y_one_hot' - W_softmax)) + lambda_B * B;
    B = B - alpha * g_B;
    B = max(B, 0);

    % Compute the loss
    reconstruction_loss = sum(sum((X_train - WH).^2));
    classification_loss = -sum(sum(Y_one_hot' .* log(W_softmax + 1e-9)));
    class_regularization_loss = 0.5 * lambda_B * sum(sum(B .^ 2));
    total_loss = reconstruction_loss + gamma * classification_loss + class_regularization_loss;

    % performance check with test dataset
    if iter >= 5 && mod(iter,chk_period) == 0
        W_test = nan(size(X_test, 1),k);
        for test_ii = 1:size(X_test, 1)
            W_test(test_ii, :) = lsqnonneg(H', X_test(test_ii, :)', lsq_options)';
        end

        pred_classes_integ = classify_samples_0504(W_test, B);
        test_acc_history(iter)=sum(pred_classes_integ==Y_test') / length(Y_test');

        [temp_b,~,~] = mnrfit(W,Y_train,IterationLimit=500);
        temp_prob = mnrval(temp_b,W_test);

        pred_class_twoStep = nan(length(temp_prob),1);
        for ii_test=1:length(temp_prob)
            [~, pred_class_twoStep(ii_test)] = max(temp_prob(ii_test,:));
        end

        test_acc=test_acc_history(~isnan(test_acc_history));
        if test_acc(end) >= max(test_acc)
            W_best = W;
            H_best = H;
            B_best = B;
            pred_integrated=pred_classes_integ;
            pred_two_step=pred_class_twoStep;
        end
    end

    % Update logs
    loss_history(iter) = total_loss;

    if gamma == 0
        if total_loss <= min(loss_history)
            W_best = W;
            H_best = H;
            B_best = B;
            pred_integrated=pred_classes_integ;
            pred_two_step=pred_class_twoStep;
        end
    end

    % Check convergence
    if gamma == 0
        if iter - find(loss_history == min(loss_history), 1, 'first') > 50
            disp('break bc no further improvement')
            disp(max(test_acc_history(~isnan(test_acc_history))))
            break;
        end
    else
        test_acc = test_acc_history(~isnan(test_acc_history));
        if ~isempty(test_acc)
            if (iter > 100 && test_acc(end) < 0.9 * max(test_acc)) || ...
                    (iter > 100 && test_acc(end) <= test_acc(max(1,end-10)))
                disp('break bc no further improvement')
                disp(max(test_acc))
                break;
            end
        end
    end

    best_acc=max(test_acc_history(~isnan(test_acc_history)));
end

end

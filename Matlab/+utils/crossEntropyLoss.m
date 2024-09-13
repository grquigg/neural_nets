function loss = crossEntropyLoss(expected, actual)
    epsilon = 1e-15;
    actual = max(epsilon, min(1-epsilon, actual));
    
    % Calculate the cross-entropy loss
    N = size(expected, 1); % Number of samples
    loss = -sum(sum(expected .* log(actual)));
end
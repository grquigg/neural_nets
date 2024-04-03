function loss = crossEntropyLoss(expected, actual)
    loss = sum(log(actual - expected));
end
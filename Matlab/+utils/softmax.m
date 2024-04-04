function probs = softmax(arr)
     logits = exp(arr);
     total = log(sum(logits, 2));
     probs = exp(arr - total);
end
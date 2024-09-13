function probs = softmax(arr)
     
     arr = arr - max(arr, [], 2);
     logits = exp(arr);
     total = sum(logits, 2);
     probs = logits ./ total;
end
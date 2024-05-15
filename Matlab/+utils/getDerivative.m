
function deriv = getDerivative(func)
    if(isequal(func, @utils.relu))
        deriv = @utils.reluDerivative;
    elseif(isequal(func, @utils.sigmoid))
        deriv = @utils.sigmoidDerivative;
    end
end
% sigmoid function
function z = sigmoid(t)
    z = 1 ./ (1 + e.^-t);
end
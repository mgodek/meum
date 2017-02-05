% sigmoid function
function z = sigmoid(t)
    z = 1 ./ (1 + e.^-t);
    %z = max(0,t);
end
% gradient of sigmoid function
function z = sigmoidGradient(t)
    z = t .* (1- t);
end

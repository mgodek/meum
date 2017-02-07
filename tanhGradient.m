% gradient of softplus function
function z = softplusGradient(t)
    z = 1 - tanh(t) .* tanh(t);
end

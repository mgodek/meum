% gradient of tanh function
function z = tanhGradient(t)
    z = 1 - tanh(t) .* tanh(t);
end

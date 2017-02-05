% gradient of softplus function
function z = softplusGradient(t)
    z = (e.^t)/(e.^t+1);
end

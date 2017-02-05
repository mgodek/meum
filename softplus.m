% activation function
function z = softplus(t)
    z = log(1+e.^t);
end
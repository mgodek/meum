% preduction function
function [outL a2]= predict(tstv, theta1, theta2)
    a1 = sigmoid(tstv * theta1);
    a2 = sigmoid(a1 * theta2);
    [rv ri] = max(a2);
    outL = ri;
end
% preduction function
function [y]= predict(tstv, theta)
    a = cell(1, rows(theta));
    a{1} = sigmoid(tstv * theta{1});
    a{2} = sigmoid(a{1} * theta{2});
    y = a{end};
end
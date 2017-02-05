% preduction function
function [y]= predict(actFun, tstv, theta)
    a = cell(rows(theta), 1);
    a{1} = actFun(tstv * theta{1});
    for (i = 2:rows(theta))
      a{i} = actFun(a{i-1} * theta{i});
    end
    y = a{end};
end
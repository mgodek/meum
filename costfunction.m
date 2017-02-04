% root mean square error
function e = costfunction(a1, a2)
  % 3 layers
  e = sum(sum((a1 - a2).^2))/rows(a1);
end
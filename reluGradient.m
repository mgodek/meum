% gradient of activation function
function z = reluGradient(t)
  if ( t > 0 )
    z = 1;
  else
    z = 0.01;
  endif
end

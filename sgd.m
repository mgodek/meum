% gradient descent training function which returns weights of the neural network
function [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnitsCount, hiddenLayersCount, c, epochMax)
    cinit = c;
 
    inUnitsCount = columns(tvec)
    hiddenLayersCount
    outUnitsCount = 1
    fflush(stdout);
 
    theta = cell(hiddenLayersCount + 1, 1);
    
    theta{1} = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    for (i = 2:hiddenLayersCount)
      theta{i} = ((rand(hiddenUnitsCount, hiddenUnitsCount).*2).-1);
    end
    theta{end} = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    E = zeros(1, epochMax);

    for (epoch=1:epochMax)
        if (epoch < 1000)
          c = cinit * 15;
        elseif ( epoch >= 1000 && epoch < 3000 )
          c = cinit * 7;
        else
          c = cinit;
        endif
    
        for (i = 1:rows(tvec))
            % feed forward
            a = cell(hiddenLayersCount + 2, 1);
            z = cell(hiddenLayersCount + 2, 1);
            a{1} = tvec(i,:);
            for (j = 2:hiddenLayersCount+2)
              z{j-1} = a{j-1} * theta{j-1};
              a{j} = actFun(z{j-1});
            end
            
            % expected value
            outE = zeros(1,outUnitsCount);
            outE = tlab(i);
            
            % accumulate cost
            e = costfunction(outE, a{end});
            E(epoch) += e;

            % back propagate the error
            delta = cell(hiddenLayersCount + 2, 1);
            delta{end} = ((outE-a{end}) .* actFun(a{end}))';
            for ( i = hiddenLayersCount + 1 : -1 : 2 )
              delta{i} = theta{i} * delta{i+1} .* actFunGrad(a{i})';
            end

            for ( i = 1:hiddenLayersCount + 1 )
              theta{i} = theta{i} + (c * delta{i+1} * a{i})';
            end
        end

        if ( mod(epoch,200) == 0 )
          printf('Epoch:%d \tc:%f \tCost:%f\n', epoch, c, E(epoch));
          fflush(stdout);
        endif
    end
    printf( "Epoch limit reached \n" );
    fflush(stdout);
end
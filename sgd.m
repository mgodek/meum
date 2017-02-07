% Author: Michal Godek
% gradient descent training function which returns weights of the neural network
function [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnitsCount, hiddenLayersCount, c, epochMax, errorGoal, tstv, tstl, normOfDataSet, mu)
    cinit = c;
 
    inUnitsCount = columns(tvec)
    hiddenLayersCount
    hiddenUnitsCount
    outUnitsCount = 1
    fflush(stdout);
 
    % init data structures
    theta = cell(hiddenLayersCount + 1, 1);
    
    load( "rnd_state.txt" );
    rand("state",rstate);
    
    theta{1} = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    for (i = 2:hiddenLayersCount)
      theta{i} = ((rand(hiddenUnitsCount, hiddenUnitsCount).*2).-1);
    end
    theta{end} = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    %E = zeros(1, epochMax);

    for (epoch=1:epochMax)
        % adapt learning factor
        if (epoch < epochMax*0.2)
          c = cinit * 5;
        elseif ( epoch >= epochMax*0.2 && epoch < 2*epochMax*0.2 )
          c = cinit * 2;
        elseif ( epoch >= 2*epochMax*0.2 && epoch < 3*epochMax*0.2 )
          c = cinit;
        elseif ( epoch >= 3*epochMax*0.2 && epoch < 4*epochMax*0.2 )
          c = cinit/2;
        else
          c = cinit/4;
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
            %e = costfunction(outE, a{end});
            %E(epoch) += e;

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
        
        [answer evalErr] = evaluate(tstv, tstl, actFun, theta, normOfDataSet, mu);
        
        if ( mod(epoch,epochMax*0.1) == 0 )
          printf('Epoch:%d \tc:%f \tCost:%f\n', epoch, c, evalErr);
          fflush(stdout);
        endif
        
        if (evalErr <= errorGoal)
          printf('Goal error reached %f\n', evalErr);
          fflush(stdout);
          return;
        endif
    end
    printf( "Epoch limit reached \n" );
    fflush(stdout);
end
% Author: Michal Godek
% gradient descent training function which returns weights of the neural network
function [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnitsCount, hiddenLayersCount, c, epochMax, errorGoal)
    cinit = c;
 
    inUnitsCount = columns(tvec)
    hiddenLayersCount
    hiddenUnitsCount
    outUnitsCount = 1
    fflush(stdout);
 
    % init data structures
    theta = cell(hiddenLayersCount + 1, 1);
    
    theta{1} = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    for (i = 2:hiddenLayersCount)
      theta{i} = ((rand(hiddenUnitsCount, hiddenUnitsCount).*2).-1);
    end
    theta{end} = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    E = zeros(1, epochMax);

    for (epoch=1:epochMax)
        % adapt learning factor
        if (epoch < epochMax*0.2)
          c = cinit * 20;
        elseif ( epoch >= epochMax*0.2 && epoch < 2*epochMax*0.2 )
          c = cinit * 10;
        elseif ( epoch >= 2*epochMax*0.2 && epoch < 3*epochMax*0.2 )
          c = cinit * 5;
        elseif ( epoch >= 3*epochMax*0.2 && epoch < 4*epochMax*0.2 )
          c = cinit;
        else
          c = cinit/2;
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

        if ( mod(epoch,500) == 0 )
          printf('Epoch:%d \tc:%f \tCost:%f\n', epoch, c, E(epoch)./(hiddenUnitsCount*hiddenLayersCount));
          fflush(stdout);
        endif
        
        if ((E(epoch)./(hiddenUnitsCount*hiddenLayersCount)) <= errorGoal )
          printf('Goal error reachd \n');
          fflush(stdout);
          return;
        endif
    end
    printf( "Epoch limit reached \n" );
    fflush(stdout);
end
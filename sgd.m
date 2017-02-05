% gradient descent training function which returns weights of the neural network
function [theta] = sgd(tvec, tlab, hiddenUnitsCount, hiddenLayersCount, c, epochMax)
    tvecCount = rows(tvec)
 
    cinit = c;
 
    inUnitsCount = columns(tvec)
    outUnitsCount = 1 %rows(unique(tlab))
 
    theta = cell(1, hiddenLayersCount);
    theta(1) = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    %todo theta(i)
    theta(2) = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    E = zeros(1, epochMax);

    for (epoch=1:epochMax)    

        if (epoch < 1000)
          c = cinit * 10;
        else
          c = cinit;
        endif
    
        for (i = 1:tvecCount)
            % feed forward
            a1 = tvec(i,:);
            z1 = a1 * theta{1};
            a2 = sigmoid(z1);
            z2 = a2 * theta{2};
            a3 = sigmoid(z2);
            
            % expected value
            outE = zeros(1,outUnitsCount);
            outE = tlab(i);
            
            % accumulate cost
            e = costfunction(outE, a3);
            E(epoch) += e;

            % back propagate the error
            delta3 = ((outE-a3) .* sigmoidGradient(a3))';
            delta2 = theta{2} * delta3 .* sigmoidGradient(a2)';

            theta{1} = theta{1} + (c * delta2 * a1)';
            theta{2} = theta{2} + (c * delta3 * a2)';
        end

        printf('Epoch:%d \tc:%f Cost:%f\n', epoch, c, E(epoch));
        fflush(stdout);
    end
    printf( "Epoch limit reached \n" );
    fflush(stdout);
end
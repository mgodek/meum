% gradient descent training function which returns weights of the neural network
function [theta1 theta2] = sgd(tvec, tlab, tvecVal, tlabVal, hiddenUnitsCount, c, errorMax, epochMax)
    tvecCount = rows(tvec)
    cinit = c
    cinit2 = cinit/2;
    
    inUnitsCount = columns(tvec)
    outUnitsCount = 1 %rows(unique(tlab))
 
    theta1 = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    theta2 = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    E = zeros(1, epochMax);

    for (epoch=1:epochMax)        
        for (i = 1:tvecCount)

            % feed forward
            a1 = tvec(i,:);
            z1 = a1 * theta1;
            a2 = sigmoid(z1);
            z2 = a2 * theta2;
            a3 = sigmoid(z2);
            
            % expected value
            outE = zeros(1,outUnitsCount);
            outE = tlab(i);
            
            % accumulate cost
            e = costfunction(outE, a3);
            E(epoch) += e;

            % back propagate the error
            delta3 = ((outE-a3) .* sigmoidGradient(a3))';
            delta2 = theta2 * delta3 .* sigmoidGradient(a2)';

            theta1 = theta1 + (c * delta2 * a1)';
            theta2 = theta2 + ((c * delta3 * a2)');
        end

        printf('Epoch:%d \tc:%f Cost:%f\n', epoch, c, E(epoch));
        fflush(stdout);
    end
    printf( "Epoch limit reached \n" );
    fflush(stdout);
end
% gradient descent training function which returns weights of the neural network
function [theta1 theta2] = sgd(tvec, tlab, tvecVal, tlabVal, hiddenUnitsCount, c, errorMax, epochMax)
    tvecCount = rows(tvec)
    cinit = c
    cinit2 = cinit/2;
    
    inUnitsCount = columns(tvec)
    outUnitsCount = rows(unique(tlab))
 
    theta1 = ((rand(inUnitsCount, hiddenUnitsCount).*2).-1);
    theta2 = ((rand(hiddenUnitsCount, outUnitsCount).*2).-1);

    E = zeros(1, epochMax);

    % biggest error samples
    SIZE_BIGGETS_ERROR = 300
    tvecEmax = zeros(SIZE_BIGGETS_ERROR, 2);
    
    resultMat = zeros(rows(unique(tlabVal)), rows(unique(tlabVal)));
    for (epoch=1:epochMax)
    
        % random order of using training set
        idxVec = randperm(tvecCount);
        
        if ( mode(epoch,2) == 1)
            tvecEmax = zeros(SIZE_BIGGETS_ERROR, 2);
        endif
        
        for (j = 1:tvecCount)
            i = idxVec(j);
            
            %%%%%%%%%%%%%%%% Show more often wrongly predicted examples %%%%%%%
            if ( mod(epoch, 2) == 0 )
                if ( rows(tvecEmax(tvecEmax(:,1)==i,:)) == 0 ) % skip OK example
                    continue
                endif
            endif
            %%%%%%%%%%%%%%%% Show more often wrongly predicted examples %%%%%%%

            % feed forward
            a1 = tvec(i,:);
            z1 = a1 * theta1;
            a2 = sigmoid(z1);
            z2 = a2 * theta2;
            a3 = sigmoid(z2);
            
            % expected value
            outE = zeros(1,outUnitsCount);
            outE(tlab(i)) = 1;

            %%% EVERY SECOND EPOCH RUN WRONGLY CLASSIFIED SAMPLES %%%%%%%
            if ( mod(epoch, 2) > 0 )
                [rv ri] = max(a3);
                if ( tlab(i) != ri )
                    [minVal minIdx] = min(tvecEmax(:,2));
                    if ( minVal < 1000 )
                        tvecEmax(minIdx,:) = [i 1000];
                    endif
                endif
            endif
            %%% EVERY SECOND EPOCH RUN WRONGLY CLASSIFIED SAMPLES %%%%%%%
            
            % accumulate cost
            e = costfunction(outE, a3);
            E(epoch) += e;

            % back propagate the error
            delta3 = ((outE-a3) .* sigmoidGradient(a3))';
            delta2 = theta2 * delta3 .* sigmoidGradient(a2)';

            theta1 = theta1 + (c * delta2 * a1)';
            theta2 = theta2 + ((c * delta3 * a2)');
        end

        %%%%%%%%%%%%%%%%% VALIDATE CURRENT RESULT %%%%%%%%%%%%%%%%%%%%%%%%%%
        resultMat = zeros(rows(unique(tlabVal)), rows(unique(tlabVal)));
        correct = 0;
        for (i=1:rows(tvecVal))
            outLab = predict(tvecVal(i,:), theta1, theta2);
            resultMat(tlabVal(i),outLab) = resultMat(tlabVal(i),outLab) + 1;
            if ( outLab == tlabVal(i) )
                correct++;
            endif
        end
        correct = correct/rows(tvecVal);
        %%%%%%%%%%%%%%%%% VALIDATE CURRENT RESULT %%%%%%%%%%%%%%%%%%%%%%%%%%

        printf('Epoch:%d \tc:%f Cost:%f\tScore:%f\n', epoch, c, E(epoch), correct);
        fflush(stdout);
        
        %%%%%%%%%%%%%%%%% ADJUST LEARNING RATE %%%%%%%%%%%%%%%%%%%%%%%%%%
        if ( 1 - correct <= errorMax*2.0 && c == cinit )
            c = c/2;
        endif
            
        if ( 1 - correct <= errorMax*1.8 && c == cinit2 )
            c = c/2;
        endif
        %%%%%%%%%%%%%%%%% ADJUST LEARNING RATE %%%%%%%%%%%%%%%%%%%%%%%%%%

        if ( 1 - correct <= errorMax )
            printf( "Success \n" );
            disp(resultMat)
            fflush(stdout);
            return
        endif
    end
    printf( "Epoch limit reached \n" );
    disp(resultMat)
    fflush(stdout);
end
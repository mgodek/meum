% Author: Michal Godek

function main(loadRstate=1, hiddenUnits=200, c=0.05, errorMax=0.028, epochMax=150)

    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif

    [tvec tlab tstv tstl] = readSets();
    % increase labels to not have 0 as index
    tlab = tlab .+ 1;
    tstl = tstl .+ 1;

    printf("PCA transform all data.\t")
    tic
    [mu trmx] = prepTransform([tvec;tstv], 44);
    tvec = pcaTransform(tvec, mu, trmx);
    tstv = pcaTransform(tstv, mu, trmx);
    toc

    ######### Randomly select samples for training and validating ##############
    tvecCount = rows(tvec);
    idxVec = randperm(tvecCount);

    tlabValCount = tvecCount/6;
    validateSet = zeros(tlabValCount, columns(tvec)+1);
    i = 1;
    for (j = 1:tlabValCount)
        validateSet(i,:) = [tlab(idxVec(j)) tvec(idxVec(j),:)];
        i += 1;
    end
    outSet = reduce_set(1.0, [0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1], validateSet);
    tlabVal = outSet(:,1);
    tvecVal = outSet(:,2:end);

    trainSet = zeros(tvecCount-tlabValCount, columns(tvec)+1);
    i = 1;
    for (j = tlabValCount+1:tvecCount)
        trainSet(i,:) = [tlab(idxVec(j)) tvec(idxVec(j),:)];
        i += 1;
    end
    outSet = reduce_set(1.0, [0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1], trainSet);
    tlab = outSet(:,1);
    tvec = outSet(:,2:end);
    ############# Randomly select samples for training and validating ##############

    tic
    [theta1 theta2] = sgd(tvec, tlab, tvecVal, tlabVal, hiddenUnits, c, errorMax, epochMax);
    toc
    fflush(stdout);

    tic
    resultMat = zeros(rows(unique(tlab)), rows(unique(tlab)));
    correct = 0;
    for (i=1:rows(tstv))
        [outLab a2]= predict(tstv(i,:), theta1, theta2);
        resultMat(tstl(i),outLab) = resultMat(tstl(i),outLab) + 1;
        if ( outLab == tstl(i) )
            correct++;
        else
            #expV = zeros(rows(unique(tstl)),1);
            #expV(tstl(i)) = 1;
            #[a2' expV]
            #[rv ri] = max(a2)
        endif
    end
    toc

    disp(resultMat)
    printf( "Result %f\n", correct/rows(tstv) );
    fflush(stdout);

endfunction
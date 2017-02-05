% Author: Michal Godek

function main(loadRstate=1, hiddenUnits=20, c=0.05, errorMax=0.028, epochMax=150)

    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif

    load "nn3-001"
    windowWidth = 3;
    a = autoreg_matrix(nn3_001, windowWidth)(windowWidth+1:end,:);
    b = [nn3_001(windowWidth+1:end,1),a(:,2:end)];
    
    tvec = b(1:50,2:end);
    tlab = b(1:50,1);
    tstv = b(51:end,2:end);
    tstl = b(51:end,1);

    tic
    tvecVal = tstv;
    tlabVal = tstl;
    [theta1 theta2] = sgd(tvec, tlab, tvecVal, tlabVal, hiddenUnits, c, errorMax, epochMax);
    toc
    fflush(stdout);

    tic
    correct = 0;
    for (i=1:rows(tstv))
        [outLab a2]= predict(tstv(i,:), theta1, theta2);
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

    printf( "Result %f\n", correct/rows(tstv) );
    fflush(stdout);

endfunction
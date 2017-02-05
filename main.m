% Author: Michal Godek

function main(loadRstate=1, hiddenUnits=60, c=0.7, errorMax=0.028, epochMax=10000)

    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif

    load "nn3-001"
    nn3_001 = nn3_001/norm(nn3_001);
    windowWidth = 7;
    a = autoreg_matrix(nn3_001, windowWidth)(windowWidth+1:end,:);
    b = [nn3_001(windowWidth+1:end,1),a(:,2:end)];
    
    tvec = b(1:50,2:end);
    tlab = b(1:50,1);
    tstv = b(51:end,2:end);
    tstl = b(51:end,1);

    tic
    [theta1 theta2] = sgd(tvec, tlab, hiddenUnits, c, errorMax, epochMax);
    toc
    fflush(stdout);

    tic
    e = 0;
    for (i=1:rows(tstv))
        [outLab]= predict(tstv(i,:), theta1, theta2);
        outLab
        tstl(i)
        e = e + costfunction(outLab, tstl(i));
    end
    toc

    e
    fflush(stdout);

endfunction
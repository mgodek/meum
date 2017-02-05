% Author: Michal Godek

function main(loadRstate=1, windowWidth=7, hiddenUnits=60, hiddenLayers=3, c=0.7, epochMax=1000)

    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif

    load "nn3-001"
    nn3_001 = nn3_001/norm(nn3_001);
    a = autoreg_matrix(nn3_001, windowWidth)(windowWidth+1:end,:);
    b = [nn3_001(windowWidth+1:end,1),a(:,2:end)];
    
    tvec = b(1:50,2:end);
    tlab = b(1:50,1);
    tstv = b(51:end,2:end);
    tstl = b(51:end,1);

    tic
    [theta] = sgd(tvec, tlab, hiddenUnits, hiddenLayers, c, epochMax);
    toc
    fflush(stdout);

    tic
    e = 0;
    for (i=1:rows(tstv))
        outLab = predict(tstv(i,:), theta)
        tstl(i)
        e = e + costfunction(outLab, tstl(i));
    end
    toc

    e
    fflush(stdout);
endfunction
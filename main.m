% Author: Michal Godek

function main(loadRstate=1, datasetName = "nn3-001", actFunName = "sigmoid", windowWidth=7, hiddenUnits=20, hiddenLayers=2, c=0.7, epochMax=10000)
    datasetName
    actFunName
    fflush(stdout);
    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif
    
    dataSet = load(datasetName);
    dataSet = dataSet/norm(dataSet);
    a = autoreg_matrix(dataSet, windowWidth)(windowWidth+1:end,:);
    b = [dataSet(windowWidth+1:end,1),a(:,2:end)];
    
    divPoint = int32(3*rows(b)/4)
    tvec = b(1:divPoint,2:end);
    tlab = b(1:divPoint,1);
    tstv = b(divPoint+1:end,2:end);
    tstl = b(divPoint+1:end,1);

    actFun = @sigmoid;
    actFunGrad = @sigmoidGradient;
    if ( strcmp("sigm",actFunName) == 1 )
      actFun = @sigmoid;
      actFunGrad = @sigmoidGradient;
    elseif ( strcmp("relu",actFunName) == 1 )
      actFun = @relu;
      actFunGrad = @reluGradient;
    else %if ( strcmp("soft",actFunName) == 1 )
      actFun = @softplus
      actFunGrad = @softplusGradient;
    endif
    
    tic
    [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnits, hiddenLayers, c, epochMax);
    toc
    fflush(stdout);

    tic
    e = 0;
    for (i=1:rows(tstv))
        outLab = predict(actFun, tstv(i,:), theta)
        tstl(i)
        e = e + costfunction(outLab, tstl(i));
    end
    toc

    e
    fflush(stdout);
endfunction
% Author: Michal Godek

function [answer e] = main(outputFile = "output", datasetName = "nn3-001", actFunName = "sigm", windowWidth=5, hiddenUnits=20, hiddenLayers=2, c=0.8, epochMax=5000, errorGoal=0.00005, biasInput = 1)

    datasetName
    actFunName
    fflush(stdout);
    
    % load and normalize data for the MLP
    dataSet = load(datasetName);
    mu = mean(dataSet);
    dataSet = dataSet .- mu;
    normOfDataSet = norm(dataSet);
    dataSet = dataSet/normOfDataSet;
    
    % change from time series to regression task
    a = autoreg_matrix(dataSet, windowWidth)(windowWidth+1:end,:);
    b = [dataSet(windowWidth+1:end,1),a(:,2:end)];
    
    % divide data to train and test set
    divPoint = int32(9*rows(b)/10)
    tvec = b(1:divPoint,2:end);
    tlab = b(1:divPoint,1);
    tstv = b(divPoint+1-windowWidth:end,2:end);
    tstl = b(divPoint+1-windowWidth:end,1);

    % pick activation function
    actFun = @sigmoid;
    actFunGrad = @sigmoidGradient;
    if ( strcmp("sigm",actFunName) == 1 )
      actFun = @sigmoid;
      actFunGrad = @sigmoidGradient;
    elseif ( strcmp("relu",actFunName) == 1 )
      actFun = @relu;
      actFunGrad = @reluGradient;
    elseif ( strcmp("tanh",actFunName) == 1 )
      actFun = @tanhActivation
      actFunGrad = @tanhGradient;
    endif
    
    % stochastic gradient descent
    tic
    [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnits, hiddenLayers, c, epochMax, errorGoal, tstv, tstl, normOfDataSet, mu, biasInput);
    toc
    fflush(stdout);

    % check on test set the predictor
    [answer e] = evaluate([tstv ones(rows(tstv).*biasInput,1)], tstl, actFun, theta, normOfDataSet, mu);

    save "-append" outputFile datasetName actFunName windowWidth hiddenUnits hiddenLayers epochMax answer e
    answer
    %theta
    e
    fflush(stdout);
endfunction
% Author: Michal Godek

function [answers,e] = main(outputFile = "output", datasetName = "nn3-001", actFunName = "sigm", windowWidth=5, hiddenUnits=20, hiddenLayers=2, c=0.8, epochMax=5000, errorGoal=0.00005)

    datasetName
    actFunName
    fflush(stdout);
    
    % load and normalize data for the MLP
    dataSet = load(datasetName);
    normOfDataSet = norm(dataSet);
    dataSet = dataSet/normOfDataSet;
    
    % change from time series to regression task
    a = autoreg_matrix(dataSet, windowWidth)(windowWidth+1:end,:);
    b = [dataSet(windowWidth+1:end,1),a(:,2:end)];
    
    % divide data to train and test set
    divPoint = int32(4*rows(b)/5)
    tvec = b(1:divPoint,2:end);
    tlab = b(1:divPoint,1);
    tstv = b(divPoint+1:end,2:end);
    tstl = b(divPoint+1:end,1);

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
    [theta] = sgd(actFun, actFunGrad, tvec, tlab, hiddenUnits, hiddenLayers, c, epochMax, errorGoal);
    toc
    fflush(stdout);

    % check on test set the predictor
    tic
    e = 0;
    answers = zeros(rows(tstl),3);
    projectedVector = zeros(rows(tstv)+1,columns(tstv));
    projectedVector(1,:) = tstv(1,:);
    for (i=1:rows(tstv))
        outLab = predict(actFun, projectedVector(i,:), theta);
        projectedVector(i+1,:) = [outLab projectedVector(i,1:end-1)];
        tstl(i);
        thisE = costfunction(outLab, tstl(i));
        e = e + thisE;
        answers(i,:) = [floor(outLab*normOfDataSet) tstl(i)*normOfDataSet thisE*normOfDataSet];
    end
    toc

    e = e * normOfDataSet;
    save "-append" outputFile datasetName actFunName windowWidth hiddenUnits hiddenLayers epochMax answers e
    answers
    e
    fflush(stdout);
endfunction
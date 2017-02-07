% Author: Michal Godek

function [answer e] = evaluate(tstv, tstl, actFun, theta, normOfDataSet, mu)
    % check on test set the predictor
    e = 0;
    answer = zeros(rows(tstl),3);
    projectedVector = zeros(rows(tstv)+1,columns(tstv));
    projectedVector(1,:) = tstv(1,:);
    for (i=1:rows(tstv))
        outLab = predict(actFun, projectedVector(i,:), theta);
        projectedVector(i+1,:) = [outLab projectedVector(i,1:end-1)];
        tstl(i);
        thisE = costfunction(outLab, tstl(i));
        e = e + thisE*normOfDataSet;
        answer(i,:) = [int32(outLab*normOfDataSet+mu) tstl(i)*normOfDataSet+mu thisE*normOfDataSet];
    end
endfunction
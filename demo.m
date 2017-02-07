% Author: Michal Godek

function demo(loadRstate=1)
    strBreak = "==========================================================================";
    tic
    datasetNames = ["nn3-001"] %;"nn3-002"]
    datasetErrorGoal = [1000;1000]
    datasetWindowWidth = [30;30]
    activationFunctions = ["sigm";"relu"]
    biasInput = 1;
    epochMax=1000
    c=0.09
    
%    PCF
%    pkg load tsa
%    
%    for ( datasetNameIdx = 1:rows(datasetNames) )
%      data = load(datasetNames(datasetNameIdx,:));
%      [parcor,sig,cil,ciu] = pacf(data',60);
%      diff = autoreg_matrix(data, 2)(1:end,2:end);
%      diff = diff(:,1) - diff(:,2);
%      diff = diff.-mean(diff);
%      diff = autoreg_matrix(diff, 2)(1:end,2:end);
%      diff = diff(:,1) - diff(:,2);
%      
%      hold all
%      plot(ciu)
%      plot(cil)
%      plot(parcor, ".")
%      input('press return to continue');
%      hold off
%      close all
%    end

    if (loadRstate == 0)
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif
    
    save outputFile rstate

    % rows: "dataset" columns "error", "windowWidth", "hiddenUnits", "hiddenLayers", "activationFunction"
    bestParams = ones(rows(datasetNames), 5).*1000000000;
    
    % param tuning
    for ( dataSetIdx = 1:rows(datasetNames) )
      for ( actFunIdx = 1:rows(activationFunctions) )
        for ( hiddenLayers=1:3 )
          for ( hiddenUnits=10:10:80 )
            printf( "%s\n\n", strBreak );
            save "-append" outputFile strBreak
            [answers, testSetError] = main("output", datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), datasetWindowWidth(dataSetIdx), hiddenUnits, hiddenLayers, c, epochMax, datasetErrorGoal(dataSetIdx), biasInput);
            if ( testSetError < bestParams(dataSetIdx, 1) )
              printf( "%s\n", strBreak );
              printf("New best %f\n", testSetError)
              bestParams(dataSetIdx, 1) = testSetError;
              bestParams(dataSetIdx, 2) = datasetWindowWidth(dataSetIdx);
              bestParams(dataSetIdx, 3) = hiddenUnits;
              bestParams(dataSetIdx, 4) = hiddenLayers;
              bestParams(dataSetIdx, 5) = actFunIdx;
              bestParams
              printf( "\n%s\n", strBreak );
              save "-append" outputFile bestParams
              fflush(stdout);
            endif
          end
          end
      end
    end

    bestParams

    printf( "%s\n execute with best params\n", strBreak );
    % execute with best params
    answers = cell(rows(datasetNames), 1);
    for ( dataSetIdx = 1:rows(datasetNames) )
      [answerVec e] = main("output", datasetNames(dataSetIdx,:), activationFunctions(bestParams(dataSetIdx, 5),:), bestParams(dataSetIdx, 2), bestParams(dataSetIdx, 3), bestParams(dataSetIdx, 4), c, epochMax, datasetErrorGoal(dataSetIdx), biasInput);
      answers{dataSetIdx} = answerVec;
    end
    
    for ( dataSetIdx = 1:rows(datasetNames) )
      dataSet = load(datasetNames(dataSetIdx,:));
      hold all
      plot(dataSet);
      plot([dataSet(1:end-rows(answers{dataSetIdx}),1);answers{dataSetIdx}(:,1)]);
      input('press return to continue');
      hold off
      close all
    end
    toc
endfunction
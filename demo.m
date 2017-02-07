% Author: Michal Godek

function demo(loadRstate=1)
    tic
    datasetNames = ["nn3-001";"nn3-085"]
    activationFunctions = ["sigm";"relu"]
    
    epochMax=5000
    c=0.8
    errorGoal = 0.00005
    
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

    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif
    
    save outputFile rstate

    % rows: "dataset" columns "error", "windowWidth", "hiddenUnits", "hiddenLayers", "activationFunction"
    bestParams = ones(rows(datasetNames), 5).*100;
    
    % param tuning
    for ( dataSetIdx = 1:rows(datasetNames) )
      for ( actFunIdx = 1:rows(activationFunctions) )
          %for ( windowWidth=10:3:16)
            windowWidth = 5;
            for ( hiddenLayers=1:2 )
              for ( hiddenUnits=20:20:60 )
                [answers, testSetError] = main("output", datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), windowWidth, hiddenUnits, hiddenLayers, c, epochMax, errorGoal);
                if ( testSetError < bestParams(dataSetIdx, 1) )
                  printf("New best %f\n", testSetError)
                  bestParams(dataSetIdx, 1) = testSetError;
                  bestParams(dataSetIdx, 2) = windowWidth;
                  bestParams(dataSetIdx, 3) = hiddenUnits;
                  bestParams(dataSetIdx, 4) = hiddenLayers;
                  bestParams(dataSetIdx, 5) = actFunIdx;
                  bestParams
                  save "-append" bestParams
                  fflush(stdout);
                endif
              end
            %end
          end
      end
    end

    bestParams

    % execute with best params
    for ( dataSetIdx = 1:rows(datasetNames) )
      [answers, e] = main("output", datasetNames(dataSetIdx,:), activationFunctions(bestParams(dataSetIdx, 5),:), bestParams(dataSetIdx, 2), bestParams(dataSetIdx, 3), bestParams(dataSetIdx, 4), c, epochMax, errorGoal);
      dataSet = load( datasetNames(dataSetIdx,:));
      hold all
      plot(dataSet);
      plot([dataSet(1:end-rows(answers),:);answers(:,1)]);
      input('press return to continue');
      hold off
      close all
    end
    toc
endfunction
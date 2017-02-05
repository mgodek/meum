% Author: Michal Godek

function demo(loadRstate=1)
    tic
    datasetNames = ["nn3-001";"nn3-085"]
    activationFunctions = ["sigm";"soft"]
    epochMax=5000

    % rows: "dataset" columns "error", "windowWidth", "hiddenUnits", "hiddenLayers", "activationFunction"
    bestParams = ones(rows(datasetNames), 5).*100;
    
    if (loadRstate == 1)
        load( "rnd_state.txt" );
        rand("state",rstate);
    else
        rstate = rand("state");
        save "rnd_state.txt" rstate
    endif
    
    save outputFile rstate
      
    for ( dataSetIdx = 1:rows(datasetNames) )
      for ( actFunIdx = 1:rows(activationFunctions) )
          for ( windowWidth=10:3:16)
            for ( hiddenLayers=1:3 )
              for ( hiddenUnits=20:20:60 )
                c=0.8;
                errorGoal = 0.00005;
                testSetError = main("output", datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), windowWidth, hiddenUnits, hiddenLayers, c, epochMax, errorGoal);
                if ( testSetError < bestParams(dataSetIdx, 1) )
                  printf("New best %f\n", testSetError)
                  bestParams(dataSetIdx, 1) = testSetError;
                  bestParams(dataSetIdx, 2) = windowWidth;
                  bestParams(dataSetIdx, 3) = hiddenUnits;
                  bestParams(dataSetIdx, 4) = hiddenLayers;
                  bestParams(dataSetIdx, 5) = actFunIdx;
                  bestParams
                  fflush(stdout);
                endif
              end
            end
          end
      end
    end
    bestParams
    toc
endfunction
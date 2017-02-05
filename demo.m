% Author: Michal Godek

function demo(loadRstate=1)
    tic
    datasetNames = ["nn3-001";"nn3-002";"nn3-051";"nn3-085"]
    activationFunctions = ["sigm";"soft";"relu"]
    epochMax=10000
    
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
          for ( windowWidth=4:2:12)
            for ( hiddenLayers=1:4 )
              for ( hiddenUnits=10:10:40 )
                c=0.8;
                errorGoal = 0.0001;
                main("output", datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), windowWidth, hiddenUnits, hiddenLayers, c, epochMax, errorGoal);
              end
            end
          end
      end
    end
    toc
endfunction
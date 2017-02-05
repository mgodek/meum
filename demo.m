% Author: Michal Godek

function demo(loadRstate=1)
    tic
    datasetNames = ["nn3-001";"nn3-085";"nn3-002";"nn3-051"]
    activationFunctions = ["sigm";"soft"] %;"relu"]
    epochMax=6000
    
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
                main("output", datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), windowWidth, hiddenUnits, hiddenLayers, c, epochMax, errorGoal);
              end
            end
          end
      end
    end
    toc
endfunction
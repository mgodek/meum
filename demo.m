% Author: Michal Godek

function demo()
    tic
    datasetNames = ["nn3-001";"nn3-002"]
    activationFunctions = ["sigm";"soft";"relu"]
    epochMax=10000
      
    for ( dataSetIdx = 1:rows(datasetNames) )
      for ( actFunIdx = 1:rows(activationFunctions) )
          for ( windowWidth=6:12)
            for ( hiddenLayers=1:2 )
              for ( hiddenUnits=4:20 )
                c=0.7;
                main("output", 1, datasetNames(dataSetIdx,:), activationFunctions(actFunIdx,:), windowWidth, hiddenUnits, hiddenLayers, c, epochMax);
              end
            end
          end
      end
    end
    toc
endfunction
% Author: Michal Godek

function demo()
    datasetNames = ["nn3-001";"nn3-002"];

    for ( dataSetIdx = 1:rows(datasetNames) )
      activationFunctions = ["sigm";"soft";"relu"]
      for ( actFunIdx = 1:rows(activationFunctions) )
        windowWidth=7;
        hiddenUnits=20;
        hiddenLayers=2;
        c=0.7;
        epochMax=10000;
        datasetName = datasetNames(dataSetIdx,:)
        actFunName = activationFunctions(actFunIdx,:)
        main(1, datasetName, actFunName, windowWidth, hiddenUnits, hiddenLayers, c, epochMax);
      end
    end

endfunction
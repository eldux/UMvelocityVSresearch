function rmse = train_MLP_LSTM(XTrain, YTrain, XValid, YValid, parameters, folder, window)
% trainLSTMx1_FCx1 builds and trains net with defined parameters for
% hyperparameter tunning for network, training options remains unchanged
%   trainLSTMx1_FCx1 net: SequenceInput>LSTM>FC->DP->FC->RegresionOutput

% Defines Network Architecture
numFeatures = size(XTrain{1},1);
numResponses = size(YTrain,2);

layers = [ ...
    sequenceInputLayer(numFeatures, 'Name', 'input')
    sequenceFoldingLayer('Name', 'fold')
    fullyConnectedLayer(parameters.fc1_units,"Name","fc_1")
    leakyReluLayer(0.1,'Name','leaky_1')
    dropoutLayer(0.2,"Name","dropout1")
    fullyConnectedLayer(parameters.fc2_units,"Name","fc_2")
    leakyReluLayer(0.1,'Name','leaky_2')
    dropoutLayer(0.2,"Name","dropout2")
    sequenceUnfoldingLayer('Name', 'unfold')
    flattenLayer('Name','flatten_1')
    lstmLayer(parameters.lstm_units,'OutputMode','last', 'Name', 'lstm_1')
    fullyConnectedLayer(numResponses, 'Name', 'fc_3')
    regressionLayer('Name','regression')];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
% Specify the training options
% Train for 60 epochs with mini-batches of size 20 using the solver 'adam'.
% Specify the learning rate 0.01. To prevent the gradients from exploding,
% set the gradient threshold to 1. To keep the sequences sorted by length,
% set 'Shuffle' to 'never'.

maxEpochs = 30;
miniBatchSize = round(32768/window);

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',20, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ... 
    'Plots','none',...
    'Verbose',0,...
    'ValidationData', {XValid, YValid},...
    'ValidationFrequency',10,...
    'OutputNetwork' ,'best-validation-loss');

% Train the Network

net = trainNetwork(XTrain,YTrain,lgraph,options);

% Test the Network

YPred = predict(net,XValid,'MiniBatchSize',miniBatchSize);

rmse = sqrt(mean((YPred(:) - YValid(:)).^2));
save(folder + "MLP-LSTM_w"+num2str(window)+'_'+num2str(parameters.fc1_units)+"_"+num2str(parameters.fc2_units)+"__"+num2str(parameters.lstm_units)+".mat", "net", "rmse")

end


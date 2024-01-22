function rmse = train_MLP_1layer(XTrain, YTrain, XValid, YValid, parameters, folder, window)
% trainLSTMx1_FCx1 builds and trains net with defined parameters for
% hyperparameter tunning for network, training options remains unchanged
%   trainLSTMx1_FCx1 net: SequenceInput>LSTM>FC->DP->FC->RegresionOutput
% This network uses different strategy

% Defines Network Architecture
numFeatures = size(XTrain,3);
numWindow = size(XTrain,2);
numResponses = size(YTrain,2);

layers = [
    imageInputLayer([1 numWindow numFeatures],"Name","seq_views_input")
    fullyConnectedLayer(parameters.fc1_units,"Name","fc_1")
    tanhLayer("Name","tanh")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(numResponses,"Name","fc_2")
    regressionLayer("Name","regressionoutput")];

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

net = trainNetwork(XTrain,YTrain,layers,options);

% Test the Network

YPred = predict(net,XValid,'MiniBatchSize',miniBatchSize);

rmse = sqrt(mean((YPred(:) - YValid(:)).^2));
save(folder + "MLP_1layer_w"+num2str(window)+"_"+num2str(parameters.fc1_units)+".mat", "net", "rmse");

end


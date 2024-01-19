function rmse = train_CNN_2layers(XTrain, YTrain, XValid, YValid, parameters, folder, window)
% trainLSTMx1_FCx1 builds and trains net with defined parameters for
% hyperparameter tunning for network, training options remains unchanged
%   trainLSTMx1_FCx1 net: SequenceInput>LSTM>FC->DP->FC->RegresionOutput
% this network uses different strategy for kernel stride selection it
% dividing the size by 2 instead of -1.

% Defines Network Architecture
numFeatures = size(XTrain,3);
numWindow = size(XTrain,2);
numResponses = size(YTrain,2);

conv1_stride = [1 floor(parameters.conv1_size/2)];
conv2_stride = [1 floor(parameters.conv2_size/2)];

layers = [
    imageInputLayer([1 numWindow numFeatures],"Name","seq_views_input")
    convolution2dLayer([1 parameters.conv1_size], parameters.conv1_units,"Name","conv_1","Padding","same", "Stride", conv1_stride)
    leakyReluLayer(0.1,'Name','leaky_1')
    convolution2dLayer([1 parameters.conv2_size], parameters.conv2_units,"Name","conv_2","Padding","same", "Stride", conv2_stride)
    leakyReluLayer(0.1,'Name','leaky_2')
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
save(folder + "conv1D_"+num2str(parameters.conv1_units)+"_"+num2str(parameters.conv1_size)+"_"+num2str(conv1_stride)+"_"+num2str(parameters.conv2_units)+"_"+num2str(parameters.conv2_size)+"_"+num2str(conv2_stride)+"_dp_fc_"+num2str(parameters.fc1_units)+".mat", "net", "rmse");

end


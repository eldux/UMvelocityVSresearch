% Clear everything before starting
clear all
% Load the trained network
load("Trained models/CNN_3layers_w19_99_3_1__101_2_1__124_3_1__200.mat");

% Set window size (that is equal w51, we can read it from structure of network)
window_size = net.Layers(1,1).InputSize(2);

% Load training and validation data
if (exist('Dataset/semi-active_dataset.mat', 'file'))
    load('Dataset/semi-active_dataset.mat')
else
    load('Dataset/semi-active_dataset_train.mat')
    load('Dataset/semi-active_dataset_valid.mat', 'XValid', 'YValid')
    load('Dataset/semi-active_dataset_test.mat', 'XTest', 'YTest')
end

%Normalize Training Predictors
mu = mean([XTrain{:}],2);
sig = std([XTrain{:}],0,2);

for i = 1:numel(XValid)
    XValid{i} = (XValid{i} - mu) ./ sig;
end

for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
end

% Prepare input windows
XValid_w = {};
XTest_w = {};
YValid_w = [];
YTest_w = [];

for i = 1:numel(XValid)
    XValid_w = [XValid_w; multi_param_splitter(XValid{i}, window_size, 1)];
    YValid_w = [YValid_w; multi_param_splitter(YValid{i}, window_size, 1)];
end

for i = 1:numel(XTest)
    XTest_w = [XTest_w; multi_param_splitter(XTest{i}, window_size, 1)];
    YTest_w = [YTest_w; multi_param_splitter(YTest{i}, window_size, 1)];
end

% Convert input windows to 4D arrays (windows over time)
numFeatures = size(XValid_w{1},1);
numWindow = size(XValid_w{1},2);
numResponses = size(YValid_w{1},1);

numSamplesValid = numel(XValid_w);
numSamplesTest = numel(XTest_w);

XValid4D = zeros(1, numWindow, numFeatures, numSamplesValid);
YValid4D = zeros(numSamplesValid, numResponses);
XTest4D = zeros(1, numWindow, numFeatures, numSamplesTest);
YTest4D = zeros(numSamplesTest, numResponses);

for i = 1:numSamplesValid
    XValid4D(1,:,:,i) = XValid_w{i}';
    YValid4D(i,:) = YValid_w{i}(:,end);
end

for i = 1:numSamplesTest
    XTest4D(1,:,:,i) = XTest_w{i}';
    YTest4D(i,:) = YTest_w{i}(:,end);
end

% Free up memory from plain windows arrays
clear XTest_w XValid_w YTest_w YValid_w

% Setup validation with average duration per sample measurement and RMSE
YTest_seq = YValid4D';
tic
YPred_seq = predict(net,XValid4D,'MiniBatchSize', 1, 'ExecutionEnvironment','cpu');
a=toc;
disp(["Duration/sample: " num2str(1000*a/numSamplesValid) "ms"])

YPred_seq = YPred_seq';
rmse_valid = sqrt(mean((YPred_seq(:) - YTest_seq(:)).^2))

% Setup testing and display RMSE
YTest_seq = YTest4D';
YPred_seq = predict(net,XTest4D,'MiniBatchSize', 1024, 'ExecutionEnvironment','auto');
YPred_seq = YPred_seq';
rmse_test = sqrt(mean((YPred_seq(:) - YTest_seq(:)).^2))
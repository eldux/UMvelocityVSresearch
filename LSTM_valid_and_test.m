% Clear everything before starting
clear all
% Load the trained network
load("Trained models/LSTM_w35_189__222.mat");

% Set window size (that is equal w51, we can read it from structure of network)
window_size = 35;

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

% Do same removal and normalization to validation set
for i = 1:numel(XValid)
    XValid_w = [XValid_w; multi_param_splitter(XValid{i}, window_size, 1)];
    YValid_w = [YValid_w; multi_param_splitter_last(YValid{i}, window_size, 1)];
end

% Do same removal and normalization to validation set
for i = 1:numel(XTest)
    XTest_w = [XTest_w; multi_param_splitter(XTest{i}, window_size, 1)];
    YTest_w = [YTest_w; multi_param_splitter_last(YTest{i}, window_size, 1)];
end

tic
YPred_seq = predict(net,XValid_w,'MiniBatchSize', 1, 'ExecutionEnvironment','cpu');
a=toc;
disp(["Duration/sample: " num2str(1000*a/size(XTest_w,1)) "ms"])

YTest_seq = YValid_w;
YTest_seq = YTest_seq';
YPred_seq = YPred_seq';

rmse_valid = sqrt(mean((YPred_seq(:) - YTest_seq(:)).^2))

tic
YPred_seq = predict(net,XTest_w,'MiniBatchSize', 1024, 'ExecutionEnvironment','auto');
a=toc;
disp(["Duration/sample: " num2str(1000*a/size(XTest_w,1)) "ms"])

YTest_seq = YTest_w;
YTest_seq = YTest_seq';
YPred_seq = YPred_seq';

rmse_test = sqrt(mean((YPred_seq(:) - YTest_seq(:)).^2))
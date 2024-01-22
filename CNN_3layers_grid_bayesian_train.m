% Clear everything before starting
clear all

addpath('Model training');

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

for i = 1:numel(XTrain)
    XTrain{i} = (XTrain{i} - mu) ./ sig;
end

for i = 1:numel(XValid)
    XValid{i} = (XValid{i} - mu) ./ sig;
end

for i = 1:numel(XTest)
    XTest{i} = (XTest{i} - mu) ./ sig;
end

window_sizes = [3 5 7 9 11 13 15 17 19 21 23 27 35 51];

for window_size = window_sizes
    clc
    clear XTrain_w XValid_w XTest_w YTrain_w YValid_w YTest_w fun results
    folder_name = "Trained models/CNN_3layers_w" + num2str(window_size) + "/"
    mkdir(folder_name)
    diary_name = folder_name + "bayesian_search_log.txt";
    diary(diary_name);
    diary on
    
    disp("Window size:"+window_size)
    XTrain_w = {};
    XValid_w = {};
    XTest_w = {};
    YTrain_w = [];
    YValid_w = [];
    YTest_w = [];

    for i = 1:numel(XTrain)
        XTrain_w = [XTrain_w; multi_param_splitter(XTrain{i}, window_size, 1)];
        YTrain_w = [YTrain_w; multi_param_splitter(YTrain{i}, window_size, 1)];
    end

    % Do same removal and normalization to validation set
    for i = 1:numel(XValid)
        XValid_w = [XValid_w; multi_param_splitter(XValid{i}, window_size, 1)];
        YValid_w = [YValid_w; multi_param_splitter(YValid{i}, window_size, 1)];
    end

    % Do same removal and normalization to validation set
    for i = 1:numel(XTest)
        XTest_w = [XTest_w; multi_param_splitter(XTest{i}, window_size, 1)];
        YTest_w = [XTest_w; multi_param_splitter(XTest{i}, window_size, 1)];
    end
    
    % Bayesian hyperparameter optimization

    % set random seed
    rng default

    % Conversion to 4D arrays
    numFeatures = size(XTrain_w{1},1);
    numWindow = size(XTrain_w{1},2);
    numResponses = size(YTrain_w{1},1);
    numSamplesTrain = numel(XTrain_w);
    numSamplesValid = numel(XValid_w);

    XTrain4D = zeros(1, numWindow, numFeatures, numSamplesTrain);
    YTrain4D = zeros(numSamplesTrain, numResponses);
    XValid4D = zeros(1, numWindow, numFeatures, numSamplesValid);
    YValid4D = zeros(numSamplesValid, numResponses);

    for i = 1:numSamplesTrain
        XTrain4D(1,:,:,i) = XTrain_w{i}';
        YTrain4D(i,:) = YTrain_w{i}(:,end);
    end

    for i = 1:numSamplesValid
        XValid4D(1,:,:,i) = XValid_w{i}';
        YValid4D(i,:) = YValid_w{i}(:,end);
    end

    clear XTest_w XTrain_w XValid_w YTest_w YTrain_w YValid_w

    % Bayesian hyperparameter optimization

    % set random seed
    rng default

    % Define tunnable parrameters
    conv1_units = optimizableVariable('conv1_units',[4, 128],'Type','integer');
    conv2_units = optimizableVariable('conv2_units',[4, 128],'Type','integer');
    conv3_units = optimizableVariable('conv3_units',[4, 128],'Type','integer');
    conv1_size = optimizableVariable('conv1_size',[2, min(window_size, 11)],'Type','integer');
    conv2_size = optimizableVariable('conv2_size',[2, min(window_size, 11)],'Type','integer');
    conv3_size = optimizableVariable('conv3_size',[2, min(window_size, 11)],'Type','integer');
    fc1_units = optimizableVariable('fc1_units',[4, 256],'Type','integer');

    fun = @(params)train_CNN_3layers(XTrain4D, YTrain4D, XValid4D, YValid4D, params, folder_name, window_size);
    results = bayesopt(fun,[conv1_units, conv1_size, conv2_units, conv2_size, conv3_units, conv3_size, fc1_units],'Verbose',1,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations', 60);

    save(folder_name + "results.mat", 'fun', 'results', 'window_size', '-v7.3')
    diary off
end
rmpath('Model training');
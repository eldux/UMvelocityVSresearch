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
   folder_name = "Trained models/BiLSTM_w" + num2str(window_size) + "/"
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
        YTrain_w = [YTrain_w; multi_param_splitter_last(YTrain{i}, window_size, 1)];
    end

    % Do same removal and normalization to validation set
    for i = 1:numel(XValid)
        XValid_w = [XValid_w; multi_param_splitter(XValid{i}, window_size, 1)];
        YValid_w = [YValid_w; multi_param_splitter_last(YValid{i}, window_size, 1)];
    end

    % Do same removal and normalization to validation set
    for i = 1:numel(XTest)
        XTest_w = [XTest_w; multi_param_splitter(XTest{i}, window_size, 1)];
        YTest_w = [XTest_w; multi_param_splitter_last(XTest{i}, window_size, 1)];
    end

    % Bayesian hyperparameter optimization

    % set random seed
    rng default

    % Define tunnable parrameters
    lstm_units = optimizableVariable('bilsmt_units',[1, 256],'Type','integer');
    fc_units = optimizableVariable('fc_units',[1, 256],'Type','integer');

    fun = @(params)train_BiLSTM(XTrain_w, YTrain_w, XValid_w, YValid_w, params, folder_name, window_size);
    results = bayesopt(fun,[lstm_units, fc_units],'Verbose',1,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations', 60);

    save(folder_name + "results.mat", 'fun', 'results', 'window_size', '-v7.3')
    diary off
end
rmpath('Model training');
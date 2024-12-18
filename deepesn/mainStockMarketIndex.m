%% Stock market index prediction by DeepESN

close all;
clear;
clc;

rng('default');

%% Load S&P500 data.
filename = 'sp500_data_2010-01-01_2018-12-31.csv';
stockMarketIdx = readtable(filename, VariableNamingRule='preserve');
closingPrice = stockMarketIdx.Close;

%% Set parameters for ESN model.
Nu = 1;    % Input dimension
Nx = 100;  % Number of nodes in the reservoir
Ny = 1;    % Output dimension
Nl = 2;    % Number of reservoir layers

inputScaling = 0.1;
networkDensity = 0.1;
spectralRadius = 0.95;
leakRate = 0.9;
interScaling = 0.1;
beta = 1e-4;  % Regularization coefficient

edim = Nu + Ny;  % Embedding dimension
delay = 1;       % Time delay

%% Set training and testing time.
learningData = delayEmbedding(closingPrice, edim, delay);

lenTrain = floor(length(learningData)*0.8);  % Training data length
lenTest = length(learningData) - lenTrain;   % Test data length

idxTestStart = lenTrain + 1;              % Test start time [pts]
idxTestEnd = idxTestStart + lenTest - 1;  % Test ending time [pts]
idxTrainStart = idxTestStart - lenTrain;  % Training start time [pts]
idxTrainEnd = idxTestStart - 1;           % Training ending time [pts]

tTrain = idxTrainStart:idxTrainEnd;
tTest  = idxTestStart:idxTestEnd;

%% Time series data for training and testing
UTrain = learningData(idxTrainStart:idxTrainEnd, 1:Nu);
DTrain = learningData(idxTrainStart:idxTrainEnd, Nu+1:end);

UTest = learningData(idxTestStart:idxTestEnd, 1:Nu);
DTest = learningData(idxTestStart:idxTestEnd, Nu+1:end);

%% Training and testing data standardization
[UTrain, muU, sigmaU] = normalize(UTrain, 1);
[DTrain, muD, sigmaD] = normalize(DTrain, 1);

UTest = (UTest - muU) ./ sigmaU;

%% ESN model
model = ESN(Nu, Nx, Ny, Nl, inputScaling, networkDensity, spectralRadius, leakRate, interScaling);

%% Training (ridge regression)
optimizer = Tikhonov(Nx, Ny, Nl, beta);
model.train(UTrain, DTrain, optimizer);

%% Model output
[XPred, YPred] = model.predict(UTest);
YPred = YPred .* sigmaD + muD;

%% Calculate RMSE
rmsePred = rmse(YPred, DTest, 2);

%% Plot
% Predict
t = tiledlayout(2, 1);
nexttile;
plot(tTest, DTest, '-', LineWidth=2.0); hold on;
plot(tTest, YPred, '--', LineWidth=2.0);
ylabel('S\&P500', Interpreter='latex');
legend('Target', 'Predict', Interpreter='latex');
xlim([idxTestStart idxTestEnd]);
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;

nexttile;
plot(tTest, rmsePred, '-', LineWidth=2.0);
ylabel('RMSE', Interpreter='latex');
xlabel('Time Step [pts]', Interpreter='latex');
xlim([idxTestStart idxTestEnd]);
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;

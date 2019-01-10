%% Train an example ConvNet to achieve very high classification, fast.
%    Load paths
clear
cd(fileparts(asnn_root))
addpath(genpath('./dlt_snn_exp_att'));
%% Load data
rng(1)
load imagenet-matconvnet-alex;
% 
% train_x = double(reshape(train_x',28,28,60000)) / 255;
% test_x = double(reshape(test_x',28,28,10000)) / 255;
% train_y = double(train_y');
% test_y = double(test_y');
% % Initialize net
% cnn.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %sub sampling layer
%     struct('type', 'c', 'outputmaps', 64, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %subsampling layer
% };
% cnn = cnnsetup(cnn, train_x, train_y);
% % Set the activation function to be a ReLU
% cnn.act_fun = @(inp)max(0, inp);
% % Set the derivative to be the binary derivative of a ReLU
% cnn.d_act_fun = @(forward_act)double(forward_act>0);
%% ReLU Train
% Set up learning constants
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs =  50;
opts.learn_bias = 0;
opts.dropout = 0.5;
cnn.first_layer_dropout = 0;
% Train - takes about 199 seconds per epoch on my machine
cnn = cnntrain(cnn, train_x, train_y, opts);
% Test
[er, bad] = cnntest(cnn, train_x, train_y);
fprintf('TRAINING Accuracy: %2.2f%%.\n', (1-er)*100);
[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Test Accuracy: %2.2f%%.\n', (1-er)*100);
%% Spike-based Testing of a ConvNet

load('mnist_cnn_99.14.mat')

t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    = 0.1;
t_opts.multip_nob   = 1;
t_opts.multip_nob_o = 0.05;
t_opts.dt           = 0.001;
t_opts.duration     = 0.04;
t_opts.report_every = 0.001;
t_opts.max_rate     =   400;
%%

cnn = convAdaptExpAttSim(cnn, test_x, test_y, t_opts);

fprintf('Done.\n');

scnn_res.threshold = t_opts.threshold;
scnn_res.multip_nob = t_opts.multip_nob;
scnn_res.duration = t_opts.duration;
scnn_res.performance = cnn.performance;
scnn_res.fr_m = cnn.fr_m;

savestring = [num2str(scnn_res.duration) 'ms_conv_snnExpATT_th_' num2str(scnn_res.threshold) '_mnob_' num2str(scnn_res.multip_nob) '.mat'];

display(savestring)
save(['./dlt_snn_exp_att/Results/MNIST/' savestring],'scnn_res');


plot(scnn_res.performance,'-')
hold on

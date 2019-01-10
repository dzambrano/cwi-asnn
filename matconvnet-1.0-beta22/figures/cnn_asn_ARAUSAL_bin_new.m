%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

rng(110)

 
%% MNIST
m_converg_time = [];
m_firing_rate = [];
m_val_accuracy = [];
m_tmv = [];
m_mean_acc = [];
m_std_acc = [];

% fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-bn-*'));
load('data/mnist-keras-bn/mnist-keras-bn-99.56_fr_56.mat');
if exist('snet') == 0
    snet = ssnet;
end

er = 0.0044;
P1er = er + 1*er/100;
m_val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
m_firing_rate = round(snet.firingrate,1);
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
m_tm = mt;

clear snet

load('data/mnist-keras-bn/mnist-keras-AROUSAL-training-0.6-0.12-50ms-99.55_fr_12.mat');
if exist('snet') == 0
    snet = ssnet;
end
m_val_accuracy = [m_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
m_firing_rate = [m_firing_rate, round(snet.firingrate,1)];
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
m_tm = [m_tm, mt];
fprintf('Selectivity %f \n', sum(snet.ara<0.5)/numel(snet.ara)*100)

%% Sonar

s_converg_time = [];
s_firing_rate = [];
s_val_accuracy = [];
s_tmv = [];
s_mean_acc = [];
s_std_acc = [];

fitDfile = dir(fullfile('data/sonar', 'sonar-*'));


er = 0.1058;
P1er = er + 1*er/100;
clear snet
load('data/sonar/sonar-89.4231_fr_24.mat');
if exist('snet') == 0
    snet = ssnet;
end


s_val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
s_firing_rate = round(snet.firingrate,1);
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
s_tm = mt;


clear snet

load('data/sonar/sonar-AR0USAL-training-1.3-0.35-50ms-89.4231_fr_10.mat');
if exist('snet') == 0
    snet = ssnet;
end
s_val_accuracy = [s_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
s_firing_rate = [s_firing_rate, round(snet.firingrate,1)];
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
s_tm = [s_tm, mt];


%% Iris
i_converg_time = [];
i_firing_rate = [];
i_val_accuracy = [];
i_tmv = [];
i_mean_acc = [];
i_std_acc = [];

fitDfile = dir(fullfile('data/iris', 'iris-*'));

er = 0.0133;
P1er = er + 1*er/100;
clear snet
load('data/iris/iris-bnth-98.6667_fr_51.mat')
if exist('snet') == 0
    snet = ssnet;
end


i_val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
i_firing_rate = round(snet.firingrate,1);
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
i_tm = mt;

clear snet

load('data/iris/iris-arousal-0.8-0.17-50ms-98.6667_fr_18.mat');
if exist('snet') == 0
    snet = ssnet;
end
i_val_accuracy = [i_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
i_firing_rate = [i_firing_rate, round(snet.firingrate,1)];
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
i_tm = [i_tm, mt];


%% Cifar
c_converg_time = [];
c_firing_rate = [];
c_val_accuracy = [];
c_tmv = [];
c_mean_acc = [];
c_std_acc = [];

fitDfile = dir(fullfile('data/cifar-keras-ycbcr', 'cifar-keras-ycbcr*'));

er = 0.101;
P1er = er + 1*er/100;
clear snet

load(fullfile(asnn_root, 'data', 'cifar', 'cifar-keras-ycbcrth0.05-89.91_fr_68.mat'))
if exist('snet') == 0
    snet = ssnet;
end
c_val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
c_firing_rate = round(snet.firingrate,1);
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
c_tm =  mt;

clear snet

load('data/cifar-keras-ycbcr/cifar-keras-arousal-training-0.15-0.05-50ms-89.94_fr_34.mat');
if exist('snet') == 0
    snet = ssnet;
end
c_val_accuracy = [c_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
c_firing_rate = [c_firing_rate, round(snet.firingrate,1)];
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
c_tm = [c_tm, mt];
fprintf('Selectivity %f \n', sum(snet.ara<0.13)/numel(snet.ara)*100)



%% Cifar 100

c1_converg_time = [];
c1_firing_rate = [];
c1_val_accuracy = [];
c1_tmv = [];
c1_mean_acc = [];
c1_std_acc = [];

fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ycbcr-*'));

er = 0.3547;
P1er = er + 1*er/100;
clear snet
load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcrth-0.03-64.69_fr_95.mat')
if exist('snet') == 0
    snet = ssnet;
end

c1_val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
c1_firing_rate = round(snet.firingrate,1);
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
c1_tm =  mt;

clear snet

load('data/cifar-100-keras-ycbcr/cifar-100-keras-arousal-training-0.15-0.03-50ms-64.66_fr_50.mat');
if exist('snet') == 0
    snet = ssnet;
end
c1_val_accuracy = [c1_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
c1_firing_rate = [c1_firing_rate, round(snet.firingrate,1)];
mt = find(sum(snet.best,2)<P1er,1);
if(isempty(mt))
    [~,mt] = min(sum(snet.best,2));
end
c1_tm = [c1_tm, mt];

fprintf('Selectivity %f \n', sum(snet.ara<0.1)/numel(snet.ara)*100)
%% ImageNet
% 
% im__converg_time = [];
% im__firing_rate = [];
% im__val_accuracy = [];
% im__tmv = [];
% im__mean_acc = [];
% im__std_acc = [];
% 
% fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ycbcr-*'));
% 
% er = 0.3547;
% P1er = er + 1*er/100;
% clear snet
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcrth-0.03-64.69_fr_95.mat')
% if exist('snet') == 0
%     snet = ssnet;
% end
% 
% im__val_accuracy = (1-sinfo_rl.val(end).top1err)*100;
% im__firing_rate = round(snet.firingrate,1);
% mt = find(sum(snet.best,2)<P1er,1);
% if(isempty(mt))
%     [~,mt] = min(sum(snet.best,2));
% end
% im__tm =  mt;
% 
% clear snet
% 
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-ARAUSAL-0.1-0.03-50ms-64.66_fr_50.mat');
% if exist('snet') == 0
%     snet = ssnet;
% end
% im__val_accuracy = [im__val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
% im__firing_rate = [im__firing_rate, round(snet.firingrate,1)];
% mt = find(sum(snet.best,2)<P1er,1);
% if(isempty(mt))
%     [~,mt] = min(sum(snet.best,2));
% end
% im__tm = [im__tm, mt];

%%


bar_fr = [m_firing_rate;
    s_firing_rate;
    i_firing_rate;
    c_firing_rate;
    c1_firing_rate];

savestring = ['data-bin-arausal-n.mat'];
save(savestring, 'bar_fr');

%%
close all
figure


c = {'MNIST','SONAR','IRIS','CIFAR10','CIFAR100'};
h = bar(bar_fr);

% 
hle = legend('Full High FR','Selective Arousal','Location','NorthWest');
ax = gca;
set(ax, ...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 10:10:100 , ...
  'xticklabel' , c, ...
  'YMinorTick'  , 'off'      , ...
  'LineWidth'   , 2         );

hYLabel = ylabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Normal');
set(ax,'FontSize',36,'FontWeight','Normal');



clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn



%%

close all





figure

subplot(4,1,1)
load('data/mnist-keras-bn/mnist-keras-bnth0.6-99.34_fr_12.mat')
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/mnist-keras-bn/mnist-keras-AROUSAL-training-0.6-0.12-50ms-99.56_fr_12.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'MNIST Low Precision','MNIST Arousal','Location','NorthEast');
legend boxoff 


h3=plot([150 150],[0 1],'LineWidth'   , 2);
er = 0.0044;

ax = gca;
set(ax, ...
    'XLim'        , [100 500],...
  'YLim'        , [0 er*5],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontName','Helvetica','FontWeight','Normal');


%%


subplot(4,1,2)
load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.15-88.78_fr_24.mat') % 0.15
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/cifar-keras-ycbcr/cifar-keras-arousal-training-0.15-0.05-50ms-89.94_fr_34.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


% hle = legend([h1, h2],'CIFAR10 Low Precision','CIFAR10 Arousal','Location','NorthEast');

h3=plot([250 250],[0 1],'LineWidth'   , 2);
% legend boxoff 
er = 0.101;
ax = gca;
set(ax, ...
    'XLim'        , [200 700],...
  'YLim'        , [0.09 er*2],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontName','Helvetica','FontWeight','Normal');

%%

subplot(4,1,3)
load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcrth-0.15-61.94_fr_21.mat') % 0.15
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/cifar-100-keras-ycbcr/cifar-100-keras-arousal-training-0.15-0.03-50ms-64.66_fr_50.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'CIFAR100 Low Precision','CIFAR100 Arousal','Location','NorthEast');
legend boxoff 

h3=plot([350 350],[0 1],'LineWidth'   , 2);
er = 0.3547;
ax = gca;
set(ax, ...
    'XLim'        , [300 700],...
  'YLim'        , [0.35 er+0.07],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontName','Helvetica','FontWeight','Normal');

%%


er = 0.370; % 62.98% - top5 85%
P1er = er + 1*er/100;

thetaRun = [0.15];


fitDfile = dir(fullfile('~/Documents/work/data/matlab/imagenet/', ['imagenet-keras-resnet18-b*-' num2str(thetaRun) '-*']));


imj_converg_time = [];

for i = 1:numel(fitDfile)
    savestring = [fitDfile(i).folder '/' fitDfile(i).name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    imj_converg_time = [imj_converg_time, sum(snet.best,2)];
    
end
sumbest = mean(imj_converg_time,2);

subplot(4,1,4)

h1=plot(sumbest,'LineWidth'   , 2);
% 
hold on
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-arousal-training-0.15-0.03-50ms-64.66_fr_50.mat');


fitDfile = dir(fullfile('~/Documents/work/data/matlab/imagenet/', ['imagenet-keras-resnet18-AROUSAL-b*-' num2str(thetaRun) '-*']));

imj_converg_time = [];

for i = 1:numel(fitDfile)
    savestring = [fitDfile(i).folder '/' fitDfile(i).name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    imj_converg_time = [imj_converg_time, sum(snet.best,2)];
    
end

sumbest = mean(imj_converg_time,2);

h2=plot(sumbest,'LineWidth'   , 2);
% 

hle = legend([h1, h2],'ImageNet Low Precision','ImageNet Arousal','Location','NorthEast');
legend boxoff 

h3=plot([350 350],[0 1],'LineWidth'   , 2);
er = 0.370;
ax = gca;
set(ax, ...
    'XLim'        , [250 1000],...
  'YLim'        , [0.35 er+0.05],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontName','Helvetica','FontWeight','Normal');
hXLabel = xlabel(ax,'Time (ms)','FontName','Helvetica','FontSize',28,'FontWeight','Normal');

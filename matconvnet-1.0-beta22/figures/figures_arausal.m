clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn



%%

close all





figure

subplot(5,1,1)
load('data/mnist-keras-bn/mnist-keras-bnth0.6-99.34_fr_12.mat')
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/mnist-keras-bn/mnist-keras-AROUSAL-training-0.6-0.12-50ms-99.56_fr_12.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'MNIST Low Precision','MNIST Arousal','Location','NorthEast');
legend boxoff 

er = 0.0044;

ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'YLim'        , [0 er*5],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontWeight','Normal');




%%

subplot(5,1,2)

load('data/sonar/sonar-bnth-83.6538_fr_10.mat') %0.85
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/sonar/sonar-ARAUSAL-0.85-0.15-50ms-90.3846_fr_14.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'SONAR Low Precision','SONAR Arousal','Location','NorthEast');
legend boxoff 
er = 0.1058;
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'YLim'        , [0.09 er*3],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontWeight','Normal');


%%
subplot(5,1,3)
load('data/iris/iris-bnth-89.3333_fr_12.mat') %0.7
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/iris/iris-ARAUSAL-0.7-0.15-50ms-98.6667_fr_20.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'IRIS Low Precision','IRSI Arousal','Location','NorthEast');
legend boxoff 
er = 0.0133;
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'YLim'        , [0.01 er*20],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontWeight','Normal');

%%


subplot(5,1,4)
load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.15-88.8_fr_24.mat') % 0.15
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/cifar-keras-ycbcr/cifar-keras-ARAUSAL-0.15-0.12-50ms-89.97_fr_32.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'CIFAR10 Low Precision','CIFAR10 Arousal','Location','NorthEast');
legend boxoff 
er = 0.101;
ax = gca;
set(ax, ...
    'XLim'        , [0 700],...
  'YLim'        , [0.1 er*2],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontWeight','Normal');

%%

subplot(5,1,5)
load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcrth-0.1-63.66_fr_30.mat') % 0.15
h1=plot(sum(ssnet.best,2),'LineWidth'   , 2);

hold on
load('data/cifar-100-keras-ycbcr/cifar-100-keras-ARAUSAL-0.1-0.03-50ms-64.66_fr_50.mat');
h2=plot(sum(ssnet.best,2),'LineWidth'   , 2);


hle = legend([h1, h2],'CIFAR100 Low Precision','CIFAR100 Arousal','Location','NorthEast');
legend boxoff 
er = 0.3547;
ax = gca;
set(ax, ...
    'XLim'        , [0 700],...
  'YLim'        , [0.35 er+0.05],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
p1=get(ax,'position');  % position of UL-most axes
set(ax,'FontSize',28,'FontWeight','Normal');

%%
% hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Normal');
% hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Normal');
% set(ax,'FontSize',36,'FontWeight','Normal');

% 
% p2=get(ax,'position'); % ditto for LR-most
% hAxOuter=axes('position',[p1(1) p2(2) p2(1)+p2(3)-p1(1)  p1(2)+p1(4)-p2(2)], ...
%               'color','none','visible','off'); % an outer axis for global p
% hX=text(-0.05,0.5,'Final spiking accuracy [%]','rotation',90,'fontsize',28, ...
%     'FontName','Calibri', ...
%         'horizontalalignment','center','verticalalignment','bottom');
% hY=text(0.5,-0.05,'Firing Rate [Hz]','rotation',0,'fontsize',28, ...
%     'FontName','Calibri', ...
%         'horizontalalignment','center','verticalalignment','top');
% 
% 


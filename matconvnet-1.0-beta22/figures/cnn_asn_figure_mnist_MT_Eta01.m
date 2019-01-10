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

spred = [];
slab = [];
testHfr = [];


er = [0.0052 0.0044 0.0054 0.0047 0.0041]; % 62.98% - top5 85%
P1er = er + er/100;

etaRun = [25 50 75 100 150];


%%
fitDfile = dir(fullfile('data/mnist-keras-bn', ['mnist-keras-eta-' num2str(25) '-0.05-*']));
    
    
savestring = [fitDfile.folder '/' fitDfile.name];
    
clear snet
    
load(savestring);
if exist('snet') == 0
    snet = ssnet;
end
            
ct = sum(snet.best,2);
m_converg_time = [m_converg_time, ct(1:500)];
m_firing_rate = [m_firing_rate, round(snet.firingrate,1)];
m_val_accuracy = [m_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(m_converg_time(:)<P1er(1),1);
if(isempty(tm))
    [~,tm] = min(m_converg_time(:));
end
m_tmv = [m_tmv, tm];
m_mean_acc = [m_mean_acc, (1-mean(m_converg_time(tm:end)))];
m_std_acc = [m_std_acc, std(m_converg_time(tm:end))];



%%



for j = 1:size(etaRun,2)    
    

    fitDfile = dir(fullfile('data/mnist-keras-bn', ['mnist-keras-eta-' num2str(etaRun(j)) '-0.1-*']));
    
    
    savestring = [fitDfile.folder '/' fitDfile.name];
    
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
            
    ct = sum(snet.best,2);
    m_converg_time = [m_converg_time, ct(1:500)];
    m_firing_rate = [m_firing_rate, round(snet.firingrate,1)];
    m_val_accuracy = [m_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(m_converg_time(:)<P1er(j),1);
    if(isempty(tm))
        [~,tm] = min(m_converg_time(:));
    end
    m_tmv = [m_tmv, tm];
    m_mean_acc = [m_mean_acc, (1-mean(m_converg_time(tm:end)))];
    m_std_acc = [m_std_acc, std(m_converg_time(tm:end))];

    
end

[m_sortfr, m_indfr] = sort(m_firing_rate);



%%
close all
figure
subplot(1,2,1)
% h1=shadedErrorBar(m_sortfr,m_mean_acc(m_indfr)*100,m_std_acc(m_indfr)*100,'b');
h1=plot(m_converg_time,'LineWidth',2);

% hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine, h5.mainLine],'MNIST','SONAR','IRIS','CIFAR','CIFAR100','Location','SouthEast');
hle = legend('25_{hp}', '25', '50', '75', '100', '150','Location','NorthEast');
legend boxoff
ax = gca;
set(ax, ...
    'XLim'        , [0 500],...
  'YLim'        , [0 1],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 0:.50:1 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Classification Error','FontName','Helvetica','FontSize',36,'FontWeight','Normal');
hXLabel = xlabel(ax,'Time (ms)','FontName','Helvetica','FontSize',36,'FontWeight','Normal');
set(ax,'FontSize',36,'FontWeight','Normal');


subplot(1,2,2)
% h1=shadedErrorBar(m_sortfr,m_mean_acc(m_indfr)*100,m_std_acc(m_indfr)*100,'b');
h2=plot([25],m_firing_rate(1),'d','MarkerSize',16,'MarkerFaceColor',[0 0.447 0.741]);
hold on
h1=plot(etaRun,wrev(m_sortfr(1:end-1)),'-o','LineWidth',2);
% hold on


ax = gca;

set(ax, ...
    'XLim'        , [0 175],...
  'YLim'        , [0 250],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , etaRun , ...
  'YTick'       , 0:50:250 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );

hle = legend([h2],'25_{hp}','Location','NorthEast');
legend boxoff
hYLabel = ylabel(ax,'Firing Rate (Hz)','FontName','Helvetica','FontSize',36,'FontWeight','Normal');
hXLabel = xlabel(ax,'\tau_{\eta} (ms)','FontName','Helvetica','FontSize',36,'FontWeight','Normal');
set(ax,'FontSize',36,'FontWeight','Normal');



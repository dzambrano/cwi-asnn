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

fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-bn-9*'));

er = 0.0044;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name; 
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
        
        
    m_converg_time = [m_converg_time, sum(snet.best,2)];
    m_firing_rate = [m_firing_rate, round(snet.firingrate,1)];
    m_val_accuracy = [m_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(m_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(m_converg_time(:,i));
    end
    m_tmv = [m_tmv, tm];
    m_mean_acc = [m_mean_acc, (1-mean(m_converg_time(tm:end,i)))];
    m_std_acc = [m_std_acc, std(m_converg_time(tm:end,i))];
end

[m_sortfr, m_indfr] = sort(m_firing_rate);



%%
close all
figure

% h1=shadedErrorBar(m_sortfr,m_mean_acc(m_indfr)*100,m_std_acc(m_indfr)*100,'b');
h1=plot(m_sortfr,m_tmv(m_indfr),'b','LineWidth',2);
hold on
% plot(1:150,99.56*ones(150,1),'--b');

% 
% h2=shadedErrorBar(s_sortfr,s_mean_acc(s_indfr)*100,s_std_acc(s_indfr)*100,'m');
% hold on
% plot(1:150,89.42*ones(150,1),'--m');
% 
% 
% h3=shadedErrorBar(i_sortfr,i_mean_acc(i_indfr)*100,i_std_acc(i_indfr)*100,'r');
% hold on
% plot(1:150,98.67*ones(150,1),'--r');
% 
% 
% h4=shadedErrorBar(c_sortfr,c_mean_acc(c_indfr)*100,c_std_acc(c_indfr)*100,'c');
% hold on
% plot(1:150,89.95*ones(150,1),'--c');
% 
% 
% h5=shadedErrorBar(c1_sortfr,c1_mean_acc(c1_indfr)*100,c1_std_acc(c1_indfr)*100,'g');
% hold on
% plot(1:150,64.53*ones(150,1),'--g');

% 
% set(get(get(h1.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h1.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

% set(get(get(h2.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h2.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h3.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h3.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h4.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h4.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h5.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h5.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');


% hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine, h5.mainLine],'MNIST','SONAR','IRIS','CIFAR','CIFAR100','Location','SouthEast');
hle = legend(h1,'MNIST','Location','SouthEast');
ax = gca;
set(ax, ...
    'XLim'        , [0 100],...
  'YLim'        , [100 500],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 100:50:500 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Matching Time (ms) ','FontName','Helvetica','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Firing Rate (Hz)','FontName','Helvetica','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',36,'FontWeight','Bold');



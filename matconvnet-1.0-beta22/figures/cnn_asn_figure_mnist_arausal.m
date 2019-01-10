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

fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-bn-*'));

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
 
%% MNIST
ma_converg_time = [];
ma_firing_rate = [];
ma_val_accuracy = [];
ma_tmv = [];
ma_mean_acc = [];
ma_std_acc = [];

ma_sele = [];

fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-ARAUSAL-*'));

er = 0.0044;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name; 
    clear snet
    
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
        
        
    ma_converg_time = [ma_converg_time, sum(snet.best,2)];
    ma_firing_rate = [ma_firing_rate, round(snet.firingrate,1)];
    ma_val_accuracy = [ma_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(ma_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(ma_converg_time(:,i));
    end
    ma_tmv = [ma_tmv, tm];
    ma_mean_acc = [ma_mean_acc, (1-mean(ma_converg_time(tm:end,i)))];
    ma_std_acc = [ma_std_acc, std(ma_converg_time(tm:end,i))];
    if isfield(snet,'ara')
        ma_sele = [ma_sele, sum(snet.ara<0.5)/numel(snet.ara)*100];
    else
        ma_sele = [ma_sele, 0];
    end
end

[ma_sortfr, ma_indfr] = sort(ma_firing_rate);


% %% MNIST
% ma12_converg_time = [];
% ma12_firing_rate = [];
% ma12_val_accuracy = [];
% ma12_tmv = [];
% ma12_mean_acc = [];
% ma12_std_acc = [];
% 
% ma12_sele = [];
% 
% fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-ARAUSAL-1.2*'));
% 
% er = 0.0044;
% P1er = er + 1*er/100;
% 
% for i = 1:numel(fitDfile)
%         
%     savestring = fitDfile(i).name; 
%     clear snet
%     
%     load(savestring);
%     if exist('snet') == 0
%         snet = ssnet;
%     end
%         
%         
%     ma12_converg_time = [ma12_converg_time, sum(snet.best,2)];
%     ma12_firing_rate = [ma12_firing_rate, round(snet.firingrate,1)];
%     ma12_val_accuracy = [ma12_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
%     tm = find(ma12_converg_time(:,i)<P1er,1);
%     if(isempty(tm))
%         [~,tm] = min(ma12_converg_time(:,i));
%     end
%     ma12_tmv = [ma12_tmv, tm];
%     ma12_mean_acc = [ma12_mean_acc, (1-mean(ma12_converg_time(tm:end,i)))];
%     ma12_std_acc = [ma12_std_acc, std(ma12_converg_time(tm:end,i))];
%     if isfield(snet,'ara')
%         ma12_sele = [ma12_sele, sum(snet.ara<0.9)/numel(snet.ara)*100];
%     else
%         ma12_sele = [ma12_sele, 0];
%     end
%     
% end
% 
% [ma12_sortfr, ma12_indfr] = sort(ma12_firing_rate);

% %% MNIST
% ma12_1s_converg_time = [];
% ma12_1s_firing_rate = [];
% ma12_1s_val_accuracy = [];
% ma12_1s_tmv = [];
% ma12_1s_mean_acc = [];
% ma12_1s_std_acc = [];
% 
% ma12_1s_sele = [];
% 
% fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-1s-keras-ARAUSAL-1.2*'));
% 
% er = 0.0044;
% P1er = er + 1*er/100;
% 
% for i = 1:numel(fitDfile)
%         
%     savestring = fitDfile(i).name; 
%     clear snet
%     
%     load(savestring);
%     if exist('snet') == 0
%         snet = ssnet;
%     end
%         
%         
%     ma12_1s_converg_time = [ma12_1s_converg_time, sum(snet.best,2)];
%     ma12_1s_firing_rate = [ma12_1s_firing_rate, round(snet.firingrate,1)];
%     ma12_1s_val_accuracy = [ma12_1s_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
%     tm = find(ma12_1s_converg_time(:,i)<P1er,1);
%     if(isempty(tm))
%         [~,tm] = min(ma12_1s_converg_time(:,i));
%     end
%     ma12_1s_tmv = [ma12_1s_tmv, tm];
%     ma12_1s_mean_acc = [ma12_1s_mean_acc, (1-mean(ma12_1s_converg_time(tm:end,i)))];
%     ma12_1s_std_acc = [ma12_1s_std_acc, std(ma12_1s_converg_time(tm:end,i))];
%     if isfield(snet,'ara')
%         ma12_1s_sele = [ma12_1s_sele, sum(snet.ara<0.9)/numel(snet.ara)*100];
%     else
%         ma12_1s_sele = [ma12_1s_sele, 0];
%     end
%     
% end
% 
% [ma12_1s_sortfr, ma12_1s_indfr] = sort(ma12_1s_firing_rate);

%%
close all
figure

h1=shadedErrorBar(m_sortfr,m_mean_acc(m_indfr)*100,m_std_acc(m_indfr)*100,'b');
hold on

h2=shadedErrorBar(ma_sortfr,ma_mean_acc(ma_indfr)*100,ma_std_acc(ma_indfr)*100,'g');

% h3=shadedErrorBar(ma12_sortfr,ma12_mean_acc(ma12_indfr)*100,ma12_std_acc(ma12_indfr)*100,'--r');
% 
% h4=shadedErrorBar(ma12_1s_sortfr,ma12_1s_mean_acc(ma12_1s_indfr)*100,ma12_1s_std_acc(ma12_1s_indfr)*100,'--r');

fill([ma_sortfr,fliplr(m_sortfr(21:end-4))], [ma_mean_acc(ma_indfr)*100, fliplr(m_mean_acc(m_indfr(21:end-4))*100)],'g');
alpha(0.20)

plot(1:150,99.56*ones(150,1),'--b');



set(get(get(h1.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h1.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h2.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

% set(get(get(h3.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h3.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% % 
% set(get(get(h4.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h4.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h5.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h5.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');


% hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine, h5.mainLine],'MNIST','SONAR','IRIS','CIFAR','CIFAR100','Location','SouthEast');
% hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine],'MNIST-SNN','MNIST-ARAUSAL 1.0','MNIST-ARAUSAL 1.2','MNIST-ARAUSAL 1.2 1s','Location','SouthEast');
hle = legend([h1.mainLine, h2.mainLine],'MNIST-SNN','MNIST-ARAUSAL','Location','SouthEast');
ax = gca;
set(ax, ...
    'XLim'        , [0 60],...
  'YLim'        , [90 100],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 90:2:100 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',36,'FontWeight','Bold');

%% Figure 2

close all
figure
subplot(1,2,1), h1=plot(ma_sele(ma_indfr),ma_mean_acc(ma_indfr)*100,'-*b','LineWidth', 2 );

hold on
% h2=plot(ma12_sele(ma12_indfr),ma12_mean_acc(ma12_indfr)*100,'-*r','LineWidth', 2 );

ax = gca;
set(ax, ...
    'XLim'        , [0 100],...
  'YLim'        , [96 100],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:20:100 , ...
  'YTick'       , 96:2:100 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',36,'FontWeight','Bold');
hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Selectivity [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');


% hle = legend([h1, h2],'MNIST-ARAUSAL 1.0','MNIST-ARAUSAL 1.2','Location','SouthEast');




% ma_sele(end) = 100;
% subplot(1,2,2), h1=plot(ma_sele(ma_indfr),ma_sortfr,'-*b','LineWidth', 2 );
subplot(1,2,2), h1=plot(ma_sele(ma_indfr),ma_sortfr(ma_indfr),'-*b','LineWidth', 2 );

hold on
% h2=plot(ma12_sele(ma12_indfr),ma12_sortfr,'-*r','LineWidth', 2 );
ax = gca;
set(ax, ...
    'XLim'        , [0 100],...
  'YLim'        , [0 60],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:20:100 , ...
  'YTick'       , 0:10:100 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',36,'FontWeight','Bold');
hYLabel = ylabel(ax,'Average Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Selectivity [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');


% hle = legend([h1, h2],'MNIST-ARAUSAL 1.0','MNIST-ARAUSAL 1.2','Location','SouthEast');

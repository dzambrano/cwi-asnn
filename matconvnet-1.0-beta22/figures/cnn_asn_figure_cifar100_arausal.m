%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

rng(110)


%% Cifar 100

c1_converg_time = [];
c1_firing_rate = [];
c1_val_accuracy = [];
c1_tmv = [];
c1_mean_acc = [];
c1_std_acc = [];

fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ycbcr*'));

er = 0.3547;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;    
    load(savestring);
    snet = ssnet;
    c1_converg_time = [c1_converg_time, sum(snet.best,2)];
    c1_firing_rate = [c1_firing_rate, round(snet.firingrate,1)];
    c1_val_accuracy = [c1_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(c1_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(c1_converg_time(:,i));
    end
    c1_tmv = [c1_tmv, tm];
    c1_mean_acc = [c1_mean_acc, (1-mean(c1_converg_time(tm:end,i)))];
    c1_std_acc = [c1_std_acc, std(c1_converg_time(tm:end,i))];
end

[c1_sortfr, c1_indfr] = sort(c1_firing_rate);
 
%% Cifar 100
% 
% c1a2_converg_time = [];
% c1a2_firing_rate = [];
% c1a2_val_accuracy = [];
% c1a2_tmv = [];
% c1a2_mean_acc = [];
% c1a2_std_acc = [];
% 
% fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ARAUSAL-0.2-*'));
% 
% er = 0.3547;
% P1er = er + 1*er/100;
% 
% for i = 1:numel(fitDfile)
%         
%     savestring = fitDfile(i).name;    
%     load(savestring);
%     snet = ssnet;
%     c1a2_converg_time = [c1a2_converg_time, sum(snet.best,2)];
%     c1a2_firing_rate = [c1a2_firing_rate, round(snet.firingrate,1)];
%     c1a2_val_accuracy = [c1a2_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
%     tm = find(c1a2_converg_time(:,i)<P1er,1);
%     if(isempty(tm))
%         [~,tm] = min(c1a2_converg_time(:,i));
%     end
%     c1a2_tmv = [c1a2_tmv, tm];
%     c1a2_mean_acc = [c1a2_mean_acc, (1-mean(c1a2_converg_time(tm:end,i)))];
%     c1a2_std_acc = [c1a2_std_acc, std(c1a2_converg_time(tm:end,i))];
% end
% 
% [c1a2_sortfr, c1a2_indfr] = sort(c1a2_firing_rate);


%% Cifar 100

c1a25_converg_time = [];
c1a25_firing_rate = [];
c1a25_val_accuracy = [];
c1a25_tmv = [];
c1a25_mean_acc = [];
c1a25_std_acc = [];

fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ARAUSAL-0.25-*'));

er = 0.3547;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;    
    load(savestring);
    snet = ssnet;
    c1a25_converg_time = [c1a25_converg_time, sum(snet.best,2)];
    c1a25_firing_rate = [c1a25_firing_rate, round(snet.firingrate,1)];
    c1a25_val_accuracy = [c1a25_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(c1a25_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(c1a25_converg_time(:,i));
    end
    c1a25_tmv = [c1a25_tmv, tm];
    c1a25_mean_acc = [c1a25_mean_acc, (1-mean(c1a25_converg_time(tm:end,i)))];
    c1a25_std_acc = [c1a25_std_acc, std(c1a25_converg_time(tm:end,i))];
end

[c1a25_sortfr, c1a25_indfr] = sort(c1a25_firing_rate);

%% Cifar 100

c1a3_converg_time = [];
c1a3_firing_rate = [];
c1a3_val_accuracy = [];
c1a3_tmv = [];
c1a3_mean_acc = [];
c1a3_std_acc = [];

fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ARAUSAL-0.3-*'));

er = 0.3547;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;    
    load(savestring);
    snet = ssnet;
    c1a3_converg_time = [c1a3_converg_time, sum(snet.best,2)];
    c1a3_firing_rate = [c1a3_firing_rate, round(snet.firingrate,1)];
    c1a3_val_accuracy = [c1a3_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(c1a3_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(c1a3_converg_time(:,i));
    end
    c1a3_tmv = [c1a3_tmv, tm];
    c1a3_mean_acc = [c1a3_mean_acc, (1-mean(c1a3_converg_time(tm:end,i)))];
    c1a3_std_acc = [c1a3_std_acc, std(c1a3_converg_time(tm:end,i))];
end

[c1a3_sortfr, c1a3_indfr] = sort(c1a3_firing_rate);

%%
close all
figure

h1=shadedErrorBar(c1_sortfr,c1_mean_acc(c1_indfr)*100,c1_std_acc(c1_indfr)*100,'g');
hold on

% h2=shadedErrorBar(c1a2_sortfr,c1a2_mean_acc(c1a2_indfr)*100,c1a2_std_acc(c1a2_indfr)*100,'b');
% 
h2=shadedErrorBar(c1a25_sortfr,c1a25_mean_acc(c1a25_indfr)*100,c1a25_std_acc(c1a25_indfr)*100,'k');
% 
h3=shadedErrorBar(c1a3_sortfr,c1a3_mean_acc(c1a3_indfr)*100,c1a3_std_acc(c1a3_indfr)*100,'r');

plot(1:150,64.53*ones(150,1),'--g');
% fill([ma_sortfr,fliplr(m_sortfr(13:end-4))], [ma_mean_acc(ma_indfr)*100, fliplr(m_mean_acc(m_indfr(13:end-4))*100)],'g');
% alpha(0.20)

% plot(1:150,99.56*ones(150,1),'--b');
% for i=1:size(ma_sortfr,2)
%     plot(ma_sortfr(i)*ones(11,1),90:100,'--r');
% end



% for i=1:size(ma12_sortfr,2)
%     plot(ma12_sortfr(i)*ones(11,1),90:100,'--r');
% end





set(get(get(h1.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h1.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h2.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h3.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h3.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h4.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h4.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% 
% set(get(get(h5.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
% set(get(get(h5.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');


% hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine, h5.mainLine],'MNIST','SONAR','IRIS','CIFAR','CIFAR100','Location','SouthEast');
hle = legend([h1.mainLine, h2.mainLine, h3.mainLine],'CIFAR100-SNN','CIFAR100-ARAUSAL 0.25','CIFAR100-ARAUSAL 0.3','Location','SouthEast');
ax = gca;
set(ax, ...
    'XLim'        , [0 60],...
  'YLim'        , [50 67],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 50:2:67 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',36,'FontWeight','Bold');




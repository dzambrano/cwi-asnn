%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

rng(110)


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

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;    
    
    load(savestring);    
    snet = ssnet;
    c_converg_time = [c_converg_time, sum(snet.best,2)];
    c_firing_rate = [c_firing_rate, round(snet.firingrate,1)];
    c_val_accuracy = [c_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(c_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(c_converg_time(:,i));
    end
    c_tmv = [c_tmv, tm];
    c_mean_acc = [c_mean_acc, (1-mean(c_converg_time(tm:end,i)))];
    c_std_acc = [c_std_acc, std(c_converg_time(tm:end,i))];
end

[c_sortfr, c_indfr] = sort(c_firing_rate);
 
%% Cifar arausal
ca_converg_time = [];
ca_firing_rate = [];
ca_val_accuracy = [];
ca_tmv = [];
ca_mean_acc = [];
ca_std_acc = [];
ca_sele = [];

fitDfile = dir(fullfile('data/cifar-keras-ycbcr', 'cifar-keras-ARAUSAL-*'));

er = 0.101;
P1er = er + 1*er/100;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;    
    
    load(savestring);    
    snet = ssnet;
    ca_converg_time = [ca_converg_time, sum(snet.best,2)];
    ca_firing_rate = [ca_firing_rate, round(snet.firingrate,1)];
    ca_val_accuracy = [ca_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(ca_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(ca_converg_time(:,i));
    end
    ca_tmv = [ca_tmv, tm];
    ca_mean_acc = [ca_mean_acc, (1-mean(ca_converg_time(tm:end,i)))];
    ca_std_acc = [ca_std_acc, std(ca_converg_time(tm:end,i))];
    
    if isfield(snet,'ara')
        ca_sele = [ca_sele, sum(snet.ara<0.2)/numel(snet.ara)*100];
    else
        ca_sele = [ca_sele, 0];
    end
    
end

[ca_sortfr, ca_indfr] = sort(ca_firing_rate);



%% Cifar
% ca25_converg_time = [];
% ca25_firing_rate = [];
% ca25_val_accuracy = [];
% ca25_tmv = [];
% ca25_mean_acc = [];
% ca25_std_acc = [];
% ca25_sele = [];
% 
% fitDfile = dir(fullfile('data/cifar-keras-ycbcr', 'cifar-keras-ARAUSAL-0.25*'));
% 
% er = 0.101;
% P1er = er + 1*er/100;
% 
% for i = 1:numel(fitDfile)
%         
%     savestring = fitDfile(i).name;    
%     
%     load(savestring);    
%     snet = ssnet;
%     ca25_converg_time = [ca25_converg_time, sum(snet.best,2)];
%     ca25_firing_rate = [ca25_firing_rate, round(snet.firingrate,1)];
%     ca25_val_accuracy = [ca25_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
%     tm = find(ca25_converg_time(:,i)<P1er,1);
%     if(isempty(tm))
%         [~,tm] = min(ca25_converg_time(:,i));
%     end
%     ca25_tmv = [ca25_tmv, tm];
%     ca25_mean_acc = [ca25_mean_acc, (1-mean(ca25_converg_time(tm:end,i)))];
%     ca25_std_acc = [ca25_std_acc, std(ca25_converg_time(tm:end,i))];
%     
%     if isfield(snet,'ara')
%         ca25_sele = [ca25_sele, sum(snet.ara<0.2)/numel(snet.ara)*100];
%     else
%         ca25_sele = [ca25_sele, 0];
%     end
%     
% end
% 
% [ca25_sortfr, ca25_indfr] = sort(ca25_firing_rate);
% 
% 
% %% Cifar
% ca3_converg_time = [];
% ca3_firing_rate = [];
% ca3_val_accuracy = [];
% ca3_tmv = [];
% ca3_mean_acc = [];
% ca3_std_acc = [];
% ca3_sele = [];
% 
% 
% 
% fitDfile = dir(fullfile('data/cifar-keras-ycbcr', 'cifar-keras-ARAUSAL-0.3*'));
% er = 0.101;
% P1er = er + 1*er/100;
% 
% for i = 1:numel(fitDfile)
%         
%     savestring = fitDfile(i).name;    
%     
%     load(savestring);    
%     snet = ssnet;
%     ca3_converg_time = [ca3_converg_time, sum(snet.best,2)];
%     ca3_firing_rate = [ca3_firing_rate, round(snet.firingrate,1)];
%     ca3_val_accuracy = [ca3_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
%     tm = find(ca3_converg_time(:,i)<P1er,1);
%     if(isempty(tm))
%         [~,tm] = min(ca3_converg_time(:,i));
%     end
%     ca3_tmv = [ca3_tmv, tm];
%     ca3_mean_acc = [ca3_mean_acc, (1-mean(ca3_converg_time(tm:end,i)))];
%     ca3_std_acc = [ca3_std_acc, std(ca3_converg_time(tm:end,i))];
%     
%     if isfield(snet,'ara')
%         ca3_sele = [ca3_sele, sum(snet.ara<0.2)/numel(snet.ara)*100];
%     else
%         ca3_sele = [ca3_sele, 0];
%     end
%     
% end
% 
% [ca3_sortfr, ca3_indfr] = sort(ca3_firing_rate);

%%
close all
figure

h1=shadedErrorBar(c_sortfr,c_mean_acc(c_indfr)*100,c_std_acc(c_indfr)*100,'c');
hold on

h2=shadedErrorBar(ca_sortfr,ca_mean_acc(ca_indfr)*100,ca_std_acc(ca_indfr)*100,'r');

% h3=shadedErrorBar(ca3_sortfr,ca3_mean_acc(ca3_indfr)*100,ca3_std_acc(ca3_indfr)*100,'g');
% 

plot(1:150,89.95*ones(150,1),'--c');
% fill([ma_sortfr,fliplr(m_sortfr(13:end-4))], [ma_mean_acc(ma_indfr)*100, fliplr(m_mean_acc(m_indfr(13:end-4))*100)],'g');
% alpha(0.20)
% 
% plot(1:150,99.56*ones(150,1),'--b');
% for i=1:size(ma_sortfr,2)
%     plot(ma_sortfr(i)*ones(11,1),90:100,'--r');
% end



% for i=1:size(ma12_sortfr,2)
%     plot(ma12_sortfr(i)*ones(11,1),90:100,'--r');
% end



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


set(get(get(h1.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h1.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h2.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
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
hle = legend([h1.mainLine, h2.mainLine],'CIFAR-SNN','CIFAR-ARAUSAL','Location','SouthEast');
ax = gca;
set(ax, ...
    'XLim'        , [0 60],...
  'YLim'        , [78 90],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 78:2:90 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',36,'FontWeight','Bold');


% fprintf('Selectivity %f \n', sum(ssnet.ara<0.2)/numel(ssnet.ara)*100)

%% Figure 2


figure
subplot(1,2,1), h1=plot(ca_sele(ca_indfr),ca_mean_acc(ca_indfr)*100,'-*r','LineWidth', 2 );

% hold on
% h2=plot(ca3_sele(ca3_indfr),ca3_mean_acc(ca3_indfr)*100,'-*g','LineWidth', 2 );
ax = gca;
set(ax, ...
    'XLim'        , [0 100],...
  'YLim'        , [78 90],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:20:100 , ...
  'YTick'       , 78:2:90 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',36,'FontWeight','Bold');
hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Selectivity [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');


% hle = legend([h1, h2],'CIFAR-ARAUSAL 0.25','CIFAR-ARAUSAL 0.3','Location','SouthEast');




% ma_sele(end) = 100;
subplot(1,2,2), h1=plot(ca_sele(ca_indfr),ca_sortfr(ca_indfr),'-*r','LineWidth', 2 );

% hold on
% h2=plot(ca3_sele(ca3_indfr),ca3_sortfr,'-*g','LineWidth', 2 );

ax = gca;
set(ax, ...
    'XLim'        , [0 100],...
  'YLim'        , [0 60],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:10:100 , ...
  'YTick'       , 0:10:100 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );
set(ax,'FontSize',36,'FontWeight','Bold');
hYLabel = ylabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Selectivity [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');


hle = legend([h1, h2],'CIFAR-ARAUSAL 0.25','CIFAR-ARAUSAL 0.3','Location','SouthEast');


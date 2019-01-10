%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

rng(110)
i=1;
 
%% MNIST
m_converg_time = [];
m_firing_rate = [];
m_val_accuracy = [];
m_tmv = [];
m_mean_acc = [];
m_std_acc = [];

fitDfile = dir(fullfile('data/mnist-keras-bn', 'mnist-keras-bn-*'));

er = 0.0044;
% er = 0.0088;
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

mmean = m_mean_acc(m_indfr);
mstd = m_std_acc(m_indfr);
mtd = m_tmv(m_indfr);


% m_indfr = 39
display(m_sortfr(39))
display(mstd(39)*100)
display(mtd(39))


%% Sonar

s_converg_time = [];
s_firing_rate = [];
s_val_accuracy = [];
s_tmv = [];
s_mean_acc = [];
s_std_acc = [];

fitDfile = dir(fullfile('data/sonar/sonarbnth', 'sonar-bnth*'));



er = 0.1058;
P1er = er + 1*er/100;
i = 1;

for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name;  
    clear snet
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    s_converg_time = [s_converg_time, sum(snet.best,2)];
    s_firing_rate = [s_firing_rate, round(snet.firingrate,1)];
    s_val_accuracy = [s_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(s_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(s_converg_time(:,i));
    end
    s_tmv = [s_tmv, tm];
    s_mean_acc = [s_mean_acc, (1-mean(s_converg_time(tm:end,i)))];
    s_std_acc = [s_std_acc, std(s_converg_time(tm:end,i))];
    if s_std_acc(end) < 0.0001 s_std_acc(end) = 0.0001; end

end

[s_sortfr, s_indfr] = sort(s_firing_rate);


smean = s_mean_acc(s_indfr);
sstd = s_std_acc(s_indfr);
std = s_tmv(s_indfr);


% s_indfr = 12
display(s_sortfr(12))
display(sstd(12)*100)
display(std(12))



%% Iris
i_converg_time = [];
i_firing_rate = [];
i_val_accuracy = [];
i_tmv = [];
i_mean_acc = [];
i_std_acc = [];

fitDfile = dir(fullfile('data/iris/irisbnth', 'iris-*'));

er = 0.0133;
P1er = er + 1*er/100;
i=1;
for i = 1:numel(fitDfile)
        
    savestring = fitDfile(i).name; 
    clear snet
    load(savestring);
    if exist('snet') == 0
        snet = ssnet;
    end
    
    i_converg_time = [i_converg_time, sum(snet.best,2)];
    i_firing_rate = [i_firing_rate, round(snet.firingrate,1)];
    i_val_accuracy = [i_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
    tm = find(i_converg_time(:,i)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(i_converg_time(:,i));
    end
    i_tmv = [i_tmv, tm];
    i_mean_acc = [i_mean_acc, (1-mean(i_converg_time(tm:end,i)))];        
    i_std_acc = [i_std_acc, std(i_converg_time(tm:end,i))];
%     if i_std_acc(end) < 0.0001 i_std_acc(end) = 0.0001; end
    
end

[i_sortfr, i_indfr] = sort(i_firing_rate);


imean = i_mean_acc(i_indfr);
istd = i_std_acc(i_indfr);
itd = i_tmv(i_indfr);

% s_indfr = 20
display(i_sortfr(20))
display(istd(20)*100)
display(itd(20))




%% Cifar
c_converg_time = [];
c_firing_rate = [];
c_val_accuracy = [];
c_tmv = [];
c_mean_acc = [];
c_std_acc = [];

fitDfile = dir(fullfile('data/cifar-keras-ycbcr', 'cifar-keras-ycbcr*'));

% er = 0.101;
er = .1214;
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



imean = i_mean_acc(i_indfr);
istd = i_std_acc(i_indfr);
itd = i_tmv(i_indfr);

% s_indfr = 20
% display(i_sortfr(20))
% display(istd(20)*100)
% display(itd(20))




%% Cifar 100

c1_converg_time = [];
c1_firing_rate = [];
c1_val_accuracy = [];
c1_tmv = [];
c1_mean_acc = [];
c1_std_acc = [];

fitDfile = dir(fullfile('data/cifar-100-keras-ycbcr', 'cifar-100-keras-ycbcr-*'));

er = 0.3547;
% er = 0.4487;
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


%%


savestring = ['data-arausal.mat'];
save(savestring);



%%
close all
figure

h1=shadedErrorBar(m_sortfr,m_mean_acc(m_indfr)*100,m_std_acc(m_indfr)*100,'b');
hold on
plot(1:150,99.56*ones(150,1),'--b');


h2=shadedErrorBar(s_sortfr,s_mean_acc(s_indfr)*100,s_std_acc(s_indfr)*100,'m');
hold on
plot(1:150,89.42*ones(150,1),'--m');


h3=shadedErrorBar(i_sortfr,i_mean_acc(i_indfr)*100,i_std_acc(i_indfr)*100,'r');
hold on
plot(1:150,98.67*ones(150,1),'--r');


h4=shadedErrorBar(c_sortfr,c_mean_acc(c_indfr)*100,c_std_acc(c_indfr)*100,'c');
hold on
plot(1:150,89.9*ones(150,1),'--c');


h5=shadedErrorBar(c1_sortfr,c1_mean_acc(c1_indfr)*100,c1_std_acc(c1_indfr)*100,'g');
hold on
plot(1:150,64.53*ones(150,1),'--g');


set(get(get(h1.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h1.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h2.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h3.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h3.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h4.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h4.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

set(get(get(h5.mainLine,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h5.patch,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');


hle = legend([h1.mainLine, h2.mainLine, h3.mainLine, h4.mainLine, h5.mainLine],'MNIST','SONAR','IRIS','CIFAR','CIFAR100','Location','SouthEast');
legend boxoff 
ax = gca;
set(ax, ...
    'XLim'        , [0 60],...
  'YLim'        , [40 100],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 50:10:100 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Normal');
hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Normal');
set(ax,'FontSize',36,'FontWeight','Normal');



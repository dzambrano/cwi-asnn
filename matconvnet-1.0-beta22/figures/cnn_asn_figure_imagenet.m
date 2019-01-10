%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn

rng(110)

%% load high frequency res

im_converg_time = [];
im_firing_rate = [];
im_val_accuracy = [];
im_tmv = [];
im_mean_acc = [];
im_std_acc = [];

spred = [];
slab = [];
testHfr = [];


% er = 0.3702; % 62.98% - top5 85%
er = 0.482; % 62.98% - top5 85%
P1er = er + 1*er/100;

% thetaRun = [0.05 0.1 0.15 0.2 0.3 1.25];
thetaRun = [0.3];
for j = 1:size(thetaRun,2) 
    
    fitDfile = dir(fullfile('~/Documents/work/data/matlab/imagenet/', ['imagenet-keras-resnet18-b*-' num2str(thetaRun(j)) '-*']));

    
    imj_converg_time = [];
    imj_firing_rate = [];
    imj_val_accuracy = [];
    
    
    
    for i = 1:numel(fitDfile)
        savestring = [fitDfile(i).folder '/' fitDfile(i).name];
        
        clear snet
        
        load(savestring);
        if exist('snet') == 0
            snet = ssnet;
        end
        
        imj_converg_time = [imj_converg_time, sum(snet.best,2)];
        imj_firing_rate = [imj_firing_rate, round(snet.firingrate,1)];
        imj_val_accuracy = [imj_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
        
        spred = [spred single(sinfo_rl.val.pred(1,:))];
        slab = [slab sinfo_rl.val.lab(1,:)];
        testHfr = [testHfr;single(ssnet.testPreSML)];
        
    end
    
    
    ser = slab == spred;
    serim = find(ser == 0);

    
    im_converg_time = [im_converg_time, mean(imj_converg_time,2)];
    im_firing_rate = [im_firing_rate, mean(snet.firingrate,1)];
    im_val_accuracy = [im_val_accuracy, (sum(ser)/50000)*100];
    tm = find(im_converg_time(:,j)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(im_converg_time(:,j));
    end
    im_tmv = [im_tmv, tm];
    im_mean_acc = [im_mean_acc, (1-mean(im_converg_time(tm:end,j)))];
    im_std_acc = [im_std_acc, std(im_converg_time(tm:end,j))];
    
end



[im_sortfr, im_indfr] = sort(im_firing_rate);


% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
% testHfr = ssnet.testPreSML;

% display(sum(ser)/50000)



%%
close all
figure

h1=shadedErrorBar(im_sortfr,im_mean_acc(im_indfr)*100,im_std_acc(im_indfr)*100,'b');
hold on
plot(1:150,62.98*ones(150,1),'--b');

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
hle = legend([h1.mainLine],'ImageNet','Location','SouthEast');
ax = gca;
set(ax, ...
    'XLim'        , [0 70],...
  'YLim'        , [52 64],...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 52:2:64 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


hYLabel = ylabel(ax,'Final spiking accuracy [%]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
hXLabel = xlabel(ax,'Firing Rate [Hz]','FontName','Calibri','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',36,'FontWeight','Bold');


save('data_imagenet', 'im_sortfr', 'im_mean_acc', 'im_indfr', 'im_std_acc');


%% load Arousal 

ima_converg_time = [];
ima_firing_rate = [];
ima_val_accuracy = [];
ima_tmv = [];
ima_mean_acc = [];
ima_std_acc = [];

spred = [];
slab = [];



er = 0.3702; % 62.98% - top5 85%
P1er = er + 1*er/100;

thetaRun = [0.15];

for j = 1:size(thetaRun,2) 
    
    fitDfile = dir(fullfile('~/Documents/work/data/matlab/imagenet/', ['imagenet-keras-resnet18t-AROUSAL-b*-' num2str(thetaRun(j)) '-*']));

    
    imj_converg_time = [];
    imj_firing_rate = [];
    imj_val_accuracy = [];
    
    
    
    for i = 1:numel(fitDfile)
        savestring = [fitDfile(i).folder '/' fitDfile(i).name];
        
        clear snet
        
        load(savestring);
        if exist('snet') == 0
            snet = ssnet;
        end
        
        imj_converg_time = [imj_converg_time, sum(snet.best,2)];
        imj_firing_rate = [imj_firing_rate, round(snet.firingrate,1)];
        imj_val_accuracy = [imj_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
        
        spred = [spred single(sinfo_rl.val.pred(1,:))];
        slab = [slab sinfo_rl.val.lab(1,:)];
        
        
    end
    
    fprintf('Selectivity %f \n', sum(snet.ara<0.1)/numel(snet.ara)*100)
    
    ser = slab == spred;
    serim = find(ser == 0);

    
    ima_converg_time = [ima_converg_time, mean(imj_converg_time,2)];
    ima_firing_rate = [ima_firing_rate, mean(snet.firingrate,1)];
    ima_val_accuracy = [ima_val_accuracy, (sum(ser)/50000)*100];
    tm = find(ima_converg_time(500:end,j)<P1er,1);
    tm =tm +500;
    if(isempty(tm))
        [~,tm] = min(ima_converg_time(:,j));
    end
    ima_tmv = [ima_tmv, tm];
    ima_mean_acc = [ima_mean_acc, (1-mean(ima_converg_time(tm:end,j)))];
    ima_std_acc = [ima_std_acc, std(ima_converg_time(tm:end,j))];
    
end



[ima_sortfr, ima_indfr] = sort(ima_firing_rate);

sum(ser)/50000



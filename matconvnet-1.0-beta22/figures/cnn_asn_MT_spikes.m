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

savestring = 'data/mnist-keras-bn/mnist-keras-testSUMSPIKES-0.12-99.5875_fr_49.mat';

er = 0.0041;
% er = 0.0088;
P1er = er + 1*er/100;


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
    [~,tm] = min(m_converg_time(:,1));
end
m_tmv = [m_tmv, tm];
m_mean_acc = [m_mean_acc, (1-mean(m_converg_time(tm:end,1)))];
m_std_acc = [m_std_acc, std(1-m_converg_time(tm:end,1))];
m_sumSpikesTime = sum(ssnet.sumSpikesTime(1:tm))/8000;

% [m_sortfr, m_indfr] = sort(m_firing_rate);

%% AROUSAL
savestring = 'data/mnist-keras-bn/mnist-keras-AROUSAL-valid-test-SUMSPIKE-0.6-0.12-50ms-99.5125_fr_11.mat';

er = 0.0041;
% er = 0.0088;
P1er = er + 1*er/100;


clear snet

load(savestring);
if exist('snet') == 0
    snet = ssnet;
end


m_converg_time = [m_converg_time, sum(snet.best,2)];
m_firing_rate = [m_firing_rate, round(snet.firingrate,1)];
m_val_accuracy = [m_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(m_converg_time(:,2)<P1er,1);
if(isempty(tm))
    [~,tm] = min(m_converg_time(:,2));
end
m_tmv = [m_tmv, tm];
m_mean_acc = [m_mean_acc, (1-mean(m_converg_time(tm:end,2)))];
m_std_acc = [m_std_acc, std(1-m_converg_time(tm:end,2))];
m_sumSpikesTime = [m_sumSpikesTime, sum(ssnet.sumSpikesTime(1:tm))/8000];


display([m_mean_acc,m_tmv,m_firing_rate])


% [m_sortfr, m_indfr] = sort(m_firing_rate);

%%
%% Sonar

s_converg_time = [];
s_firing_rate = [];
s_val_accuracy = [];
s_tmv = [];
s_mean_acc = [];
s_std_acc = [];


savestring = 'data/sonar/sonar-valid-SUMSPIKES-0.3-90.3846_fr_26.mat';

er = 0.1058;
P1er = er + 1*er/100;
i = 1;

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
s_std_acc = [s_std_acc, std(1-s_converg_time(tm:end,i))];
if s_std_acc(end) < 0.0001 s_std_acc(end) = 0.0001; end



[s_sortfr, s_indfr] = sort(s_firing_rate);

% arousal

savestring = 'data/sonar/sonar-arousal-SUMSPIKES-1.3-0.3-50ms-90.3846_fr_11.mat';

er = 0.1058;
P1er = er + 1*er/100;
i = 2;

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
s_std_acc = [s_std_acc, std(1-s_converg_time(tm:end,i))];
if s_std_acc(end) < 0.0001 s_std_acc(end) = 0.0001; end


[s_sortfr, s_indfr] = sort(s_firing_rate);


%%
%% Iris
i_converg_time = [];
i_firing_rate = [];
i_val_accuracy = [];
i_tmv = [];
i_mean_acc = [];
i_std_acc = [];


savestring = 'data/iris/iris-SUMSPIKES-0.16-98.6667_fr_48.mat';

er = 0.0133;
P1er = er + 1*er/100;
i=1;


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
i_std_acc = [i_std_acc, std(1-i_converg_time(tm:end,i))];
if i_std_acc(end) < 0.0001 i_std_acc(end) = 0.0001; end


[i_sortfr, i_indfr] = sort(i_firing_rate);

% arousal
i=2;

savestring = 'data/iris/iris-arousal-SUMSPIKES-0.8-0.16-50ms-98.6667_fr_18.mat';

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
i_std_acc = [i_std_acc, std(1-i_converg_time(tm:end,i))];
if i_std_acc(end) < 0.0001 i_std_acc(end) = 0.0001; end


[i_sortfr, i_indfr] = sort(i_firing_rate);


%% Cifar
c_converg_time = [];
c_firing_rate = [];
c_val_accuracy = [];
c_tmv = [];
c_mean_acc = [];
c_std_acc = [];

savestring = 'data/cifar-keras-valid/cifar-keras-valid-1s-SUMSPIKES-0.05-89.6625_fr_60.mat';

er = .1034;
P1er = er + 1*er/100;
        
%     savestring = fitDfile(i).name;    

load(savestring);
snet = ssnet;
c_converg_time = [c_converg_time, sum(snet.best,2)];
c_firing_rate = [c_firing_rate, round(snet.firingrate,1)];
c_val_accuracy = [c_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(c_converg_time(:,1)<P1er,1);
if(isempty(tm))
    [~,tm] = min(c_converg_time(:,1));
end
c_tmv = [c_tmv, tm];
c_mean_acc = [c_mean_acc, (1-mean(c_converg_time(tm:end,1)))];
c_std_acc = [c_std_acc, std(1-c_converg_time(tm:end,1))];
c_sumSpikesTime = sum(ssnet.sumSpikesTime(1:tm))/8000;

% [c_sortfr, c_indfr] = sort(c_firing_rate);


savestring = 'data/cifar-keras-valid/cifar-keras-arousal-valid-1s-SUMSPIKES-0.15-0.05-50ms-89.6875_fr_35.mat';

er = .1034;
P1er = er + 1*er/100;
        
%     savestring = fitDfile(i).name;    

load(savestring);
snet = ssnet;
c_converg_time = [c_converg_time, sum(snet.best,2)];
c_firing_rate = [c_firing_rate, round(snet.firingrate,1)];
c_val_accuracy = [c_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(c_converg_time(:,2)<P1er,1);
if(isempty(tm))
    [~,tm] = min(c_converg_time(:,2));
end
c_tmv = [c_tmv, tm];
c_mean_acc = [c_mean_acc, (1-mean(c_converg_time(tm:end,2)))];
c_std_acc = [c_std_acc, std(1-c_converg_time(tm:end,2))];
c_sumSpikesTime = [c_sumSpikesTime, sum(ssnet.sumSpikesTime(1:tm))/8000];


display([c_mean_acc,c_tmv,c_firing_rate])


%% Cifar 100

c1_converg_time = [];
c1_firing_rate = [];
c1_val_accuracy = [];
c1_tmv = [];
c1_mean_acc = [];
c1_std_acc = [];

savestring = 'data/cifar-100-keras-valid/cifar-100-keras-valid-1s-SUMSPIKES-0.03-63.4_fr_95.mat';


er = 0.3655;
P1er = er + 1*er/100;
        
load(savestring);
snet = ssnet;
c1_converg_time = [c1_converg_time, sum(snet.best,2)];
c1_firing_rate = [c1_firing_rate, round(snet.firingrate,1)];
c1_val_accuracy = [c1_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(c1_converg_time(:,1)<P1er,1);
if(isempty(tm))
    [~,tm] = min(c1_converg_time(:,1));
end
c1_tmv = [c1_tmv, tm];
c1_mean_acc = [c1_mean_acc, (1-mean(c1_converg_time(tm:end,1)))];
c1_std_acc = [c1_std_acc, std(1-c1_converg_time(tm:end,1))];
c1_sumSpikesTime = sum(ssnet.sumSpikesTime(1:tm))/8000;


% [c1_sortfr, c1_indfr] = sort(c1_firing_rate);
savestring = 'data/cifar-100-keras-valid/cifar-100-keras-SUMSPIKES-1.0s-arousal-valid-0.1-1x63.3875_fr_61.mat';


er = 0.3655;
P1er = er + 1*er/100;
        
load(savestring);
snet = ssnet;
c1_converg_time = [c1_converg_time, sum(snet.best,2)];
c1_firing_rate = [c1_firing_rate, round(snet.firingrate,1)];
c1_val_accuracy = [c1_val_accuracy, (1-sinfo_rl.val(end).top1err)*100];
tm = find(c1_converg_time(:,2)<P1er,1);
if(isempty(tm))
    [~,tm] = min(c1_converg_time(:,2));
end
c1_tmv = [c1_tmv, tm];
c1_mean_acc = [c1_mean_acc, (1-mean(c1_converg_time(tm:end,2)))];
c1_std_acc = [c1_std_acc, std(1-c1_converg_time(tm:end,2))];
c1_sumSpikesTime = [c1_sumSpikesTime, sum(ssnet.sumSpikesTime(1:tm))/8000];


display([c1_mean_acc,c1_tmv,c1_firing_rate])

%%

im_converg_time = [];
im_firing_rate = [];
im_val_accuracy = [];
im_tmv = [];
im_mean_acc = [];
im_std_acc = [];

spred = [];
slab = [];
testHfr = [];


er = 0.3702; % 62.98% - top5 85%
% er = 0.482; % 62.98% - top5 85%
P1er = er + 1*er/100;

% thetaRun = [0.05 0.1 0.15 0.2 0.3 1.25];
thetaRun = [0.3];
for j = 1:size(thetaRun,2) 
    
    fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-SUMSPIKE-b*-0.05*'));

    
    imj_converg_time = [];
    imj_firing_rate = [];
    imj_val_accuracy = [];
    imj_sumSpikesTime = [];
    
    
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
%         testHfr = [testHfr;single(ssnet.testPreSML)];
        imj_sumSpikesTime = [imj_sumSpikesTime; ssnet.sumSpikesTime];

        
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
    im_std_acc = [im_std_acc, std(1-im_converg_time(tm:end,j))];
    sumSpikesTime = sum(imj_sumSpikesTime,1)/50000;
    im_sumSpikesTime = sum(ssnet.sumSpikesTime(1:tm));

end


%

for j = 1:size(thetaRun,2) 
    
    fitDfile = dir(fullfile(pwd,'/data/imagenet-keras-resnet18/imagenet-keras-valid-resnet18-AROUSAL-SUMSPIKES-b*-0.15*'));

    
    imj_converg_time = [];
    imj_firing_rate = [];
    imj_val_accuracy = [];
    imj_sumSpikesTime = [];
    
    
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
%         testHfr = [testHfr;single(ssnet.testPreSML)];
        imj_sumSpikesTime = [imj_sumSpikesTime; ssnet.sumSpikesTime];
        

        
    end
    
    
    ser = slab == spred;
    serim = find(ser == 0);

    
    im_converg_time = [im_converg_time, mean(imj_converg_time,2)];
    im_firing_rate = [im_firing_rate, mean(snet.firingrate,1)];
    im_val_accuracy = [im_val_accuracy, (sum(ser)/50000)*100];
    tm = find(im_converg_time(:,2)<P1er,1);
    if(isempty(tm))
        [~,tm] = min(im_converg_time(:,2));
    end
    im_tmv = [im_tmv, tm];
    im_mean_acc = [im_mean_acc, (1-mean(im_converg_time(tm:end,2)))];
    im_std_acc = [im_std_acc, std(1-im_converg_time(tm:end,2))];
    sumSpikesTime = sum(imj_sumSpikesTime,1)/50000;
    im_sumSpikesTime = [im_sumSpikesTime, sum(ssnet.sumSpikesTime(1:tm))];
    
end

display([im_mean_acc,im_tmv,im_firing_rate])


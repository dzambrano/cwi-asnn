%% Experiment with the cnn_cifar_fc_bnorm
% clc
clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn

 

rng(110)
% vl_compilenn
%% importing keras
% ======
% % % =
% [net_rl, info_rl] = snn_cifar(...
%   'expDir', 'data/cifar-keras-valid','modelType', 'keras', 'testOnNetwork', 'cnn');
% fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);


% 89.66 matConvNet (89.76 keras) 

% exit

% savestring = 'cifar-keras.mat';
% save(savestring,'info_rl');
%%
% res_t = fixedThresholdLayersKeras(info_rl.res,net_rl);
% 
% cd data/cifar-keras-ycbcr
% 
% savestring = 'cifar-keras-ycbcr-fitDistr.mat';
% 
% save(savestring,'res_t');  
% 
%% Inference
% 
% 
% cd(asnn_root)
% addpath(genpath('./'));
% % for itheta = 0.25 : 0.05: 0.45
% for itheta = 0.05
%     [snet, sinfo_rl] = snn_cifar(...
%         'expDir', 'data/cifar-keras-valid', 'modelType', ...
%         'keras', 'testOnNetwork', 'snn', ...
%         'thresholdvalue', itheta);
%     fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
%     
%     fprintf('Mean firing rate %f Hz \n', snet.firingrate)
%     
%     
%     ssnet = {};
%     ssnet.layers = {};
%     ssnet.best = snet.best;
%     ssnet.meta = snet.meta;
%     ssnet.meanfiring = snet.meanfiring;
%     ssnet.firingrate = snet.firingrate;
%     ssnet.testPreSML = snet.testPreSML;
%     ssnet.sumSpikesTime = snet.sumSpikesTime;
% 
%     
%     for i=1:numel(snet.layers)
%         
%         if strcmp(snet.layers{i}.type,'snn_relu_th')
%             
%             ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
%                 'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
%                 'mean_spike', snet.layers{i}.mean_spike, ...
%                 'tot_spikes', snet.layers{i}.tot_spikes ) ;
% 
%         else
%             ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;
%             
%         end
%     end
%     
%     
%     
%     cd data/cifar-keras-valid
%     savestring = ['cifar-keras-train-valid-1s-sumspikes-' num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
%         '_fr_' num2str(round(snet.firingrate)) '.mat'];
%     
%     save(savestring, 'ssnet', 'sinfo_rl');
%     cd(asnn_root)
%     
%     clear snet
%     clear ssnet
%     clear sinfo_rl
%     
%     
% end
% exit

%% arousal
% 
cd(asnn_root)
addpath(genpath('./'));
% for itheta = 0.25 : 0.05: 0.45
for s_ara = 0.15

[snet, sinfo_rl] = snn_cifar(...
    'expDir', 'data/cifar-keras-valid', 'modelType', ...
    'keras', 'testOnNetwork', 'snn', ...
    'thresholdvalue', s_ara, 'arousal', 1);
fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);

fprintf('Mean firing rate %f Hz \n', snet.firingrate)
fprintf('Selectivity %f \n', sum(snet.ara<0.15)/numel(snet.ara)*100)


ssnet = {};
ssnet.layers = {};
ssnet.best = snet.best;
ssnet.meta = snet.meta;
ssnet.meanfiring = snet.meanfiring;
ssnet.firingrate = snet.firingrate;
ssnet.ara = snet.ara;
ssnet.sumSpikesTime = snet.sumSpikesTime;


for i=1:numel(snet.layers)
    
    if strcmp(snet.layers{i}.type,'snn_relu_th')
        
        ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
            'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
            'mean_spike', snet.layers{i}.mean_spike, ...
            'tot_spikes', snet.layers{i}.tot_spikes ) ;
    else
        ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;
        
    end
end


fprintf('Selectivity %f \n', sum(ssnet.ara<0.15)/numel(ssnet.ara)*100)

cd data/cifar-keras-valid
savestring = ['cifar-keras-arousal-valid-1s-SUMSPIKES-' num2str(s_ara) '-0.05-50ms-' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];

save(savestring, 'ssnet', 'sinfo_rl');
cd(asnn_root)
end

exit

%%

%% tot numb of neurons and sum spikes

sum_spikes = 0;
tot_neurons = 0;

for i=1:numel(ssnet.layers)
%     display(ssnet.layers{i}.type)
    if strcmp(ssnet.layers{i}.type,'snn_relu_th')        
        sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;   
        tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;        
    end
end


sum_spikes_im = sum_spikes/8000;
avg_firing_rate = sum_spikes_im / tot_neurons;

sum_spikes_im_b = sum(ssnet.sumSpikesTime)/8000;

sum_spikes_im_MT=sum(ssnet.sumSpikesTime(1:424))/8000;
MT_firing_rate = (sum_spikes_im_MT/tot_neurons)/0.424;


%% FIRING RATE MT
% load(fullfile(asnn_root, 'data', 'cifar-keras-valid', 'cifar-keras-valid-1s-SUMSPIKES-0.05-89.6625_fr_60.mat'))
% matching time MNIST 350

% sum_spikes_im_MT=sum(ssnet.sumSpikesTime(1:350))/8000;
% MT_firing_rate = (sum_spikes_im_MT/tot_neurons)/0.350;

% load(fullfile(asnn_root, 'data', 'cifar-keras-valid', 'cifar-keras-arousal-valid-1s-SUMSPIKES-0.15-0.05-50ms-89.6875_fr_35.mat'))
% matching time MNIST AROUSAL 441
% sum_spikes_im_MT_A=sum(ssnet.sumSpikesTime(1:441))/8000;
% MT_firing_rate_A = (sum_spikes_im_MT/tot_neurons)/0.441;


%%
%% Fan out

sum_spikes = 0;
tot_neurons = 0;
net = cnn_cifar_init_keras
fanout = zeros(1, numel(net.layers));

for i=1:numel(net.layers)
    
    if strcmp(net.layers{i}.type,'asn')        
        if strcmp(net.layers{i+1}.type,'conv')
            siz_out = size(net.layers{i+1}.weights{1});
            fanout(i)=prod(siz_out([1,2,4]));
        elseif strcmp(net.layers{i+1}.type,'pool')
            siz_out = size(net.layers{i+2}.weights{1});
            fanout(i)=prod(siz_out([1,2,4]));
        end
        
    end

end
spike_fo = 0;

for i=1:numel(ssnet.layers)
%     display(ssnet.layers{i}.type)
    if strcmp(ssnet.layers{i}.type,'snn_relu_th')        
        sum_spikes = sum_spikes + sum(double(ssnet.layers{i}.tot_spikes(:))) ;   
        tot_neurons = tot_neurons + numel(ssnet.layers{i}.tot_spikes) ;                       
        spike_fo = spike_fo + sum(double(ssnet.layers{i}.tot_spikes(:))) * fanout(i);
    end
end

spike_fo = spike_fo / 8000;


sum_spikes_im = sum_spikes/8000;
avg_firing_rate = (sum_spikes_im / tot_neurons) / 0.5;

sum_spikes_im_b = sum(ssnet.sumSpikesTime)/8000;

sum_spikes_im_MT =sum(ssnet.sumSpikesTime(1:592))/8000;
rats = sum_spikes_im_MT/sum_spikes_im;

spike_fo_MT = spike_fo * rats;




%%

sum_spikes = 0;
tot_neurons = 0;

for i=1:numel(ssnet.layers)
%     display(ssnet.layers{i}.type)
    if strcmp(ssnet.layers{i}.type,'snn_relu_th')                
        tot_neurons = tot_neurons + numel(ssnet.layers{i}.mean_spiking_im) ;        
    end
end




sum_spikes_im_b = sum(ssnet.sumSpikesTime)/8000;
avg_firing_rate = sum_spikes_im_b / tot_neurons;

%% load high frequency res
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.05-89.91_fr_68.mat')
spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
testHfr = ssnet.testPreSML;

%% load low frequency res
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.15-88.8_fr_24.mat') % 0.15
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.2-87.22_fr_18.mat') % 0.2
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.25-85.04_fr_14.mat') % 0.25
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.3-81.93_fr_12.mat') % 0.3
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.35-77.97_fr_10.mat') % 0.3
% load('data/cifar-keras-ycbcr/cifar-keras-ycbcrth0.4-72.81_fr_9.mat') % 0.3




spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim_l = find(ser == 0);
testLfr = ssnet.testPreSML;% % shiim = find(ssnet.ara < 0.9);
% av = zeros(size(serim));
% for i = 1:numel(serim)
%     
%     a = find(erim==serim(i));    
%     if isempty(a)==0
%         av(i) = a;
%     end
% end
% 
% display(sum(av>0)/numel(serim))
%     
%
falsePos = [];
fasleNeg = [];
corrToWrong = [];

for i = 1:numel(serim_l)
   
    a = find(serim==serim_l(i));
    if isempty(a)
       corrToWrong =  [corrToWrong, serim_l(i)];
    end
    
end

wrongToCorr = [];

for i = 1:numel(serim)
   
    a = find(serim_l==serim(i));
    if isempty(a)
       wrongToCorr =  [wrongToCorr, serim(i)];
    end
    
end

%% Compute threshold-line
ara = zeros(1,10000);
ara2 = zeros(1,10000);
bb = zeros(size(testLfr));
bb(:,1) = testLfr(:,1);
cc = zeros(10000,500);
dd = zeros(10000,1);

xti = 270:20:410 ;
yti = [];
for i=250:500
   
%       aa = testLfr(:,i);
      bb(:,i) = bb(:,i-1) + testLfr(:,i);
      aa = reshape(bb(:,i),10,10000);
      aa = sort(aa,'descend');
      
      % Method 1 - cumulated divergence 1st vs 2nd
      cc(:,i) = aa(1,:)-aa(2,:);     
      
      
      
      if any(i == xti)          
          eth = max(cc(corrToWrong,i));
          yti = [yti, round(eth,-1)];
%           ara(cc(:,i)<ceil(eth))=1;  
      end     
      
      
end

% figure
plot(xti,yti)
hold on

%%
p = polyfit(xti,yti+20,1);

plot(xti,xti*p(1)+p(2))

%%
%% Compute threshold-line training
ara = zeros(1,50000);
ara2 = zeros(1,50000);
bb = zeros(size(testLfr));
bb(:,1) = testLfr(:,1);
cc = zeros(50000,500);
dd = zeros(50000,1);

xti = 270:20:410 ;
yti = [];
for i=250:500
   
%       aa = testLfr(:,i);
      bb(:,i) = bb(:,i-1) + testLfr(:,i);
      aa = reshape(bb(:,i),2,50000);
      aa = sort(aa,'descend');
      
      % Method 1 - cumulated divergence 1st vs 2nd
      cc(:,i) = aa(1,:)-aa(2,:);     
      
      
      
      if any(i == xti)          
          eth = max(cc(corrToWrong,i));
          yti = [yti, round(eth,-1)];
%           ara(cc(:,i)<ceil(eth))=1;  
      end     
      
      
end

% figure
plot(xti,yti)
hold on

%%
p = polyfit(xti,yti+20,1);

plot(xti,xti*p(1)+p(2))


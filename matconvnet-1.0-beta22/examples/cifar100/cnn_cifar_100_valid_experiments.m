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

% 
% [net_rl, info_rl] = snn_cifar_100(...
%   'expDir', 'data/cifar-100-keras-valid','modelType', 'keras_valid', 'testOnNetwork', 'cnn');
% fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
% exit

% 63.45 matConvNet


% savestring = 'cifar-keras.mat';
% save(savestring,'net_rl','info_rl');
%%
% res_t = fixedThresholdLayersKeras(info_rl.res,net_rl);
% 
% cd data/cifar-100-keras-ycbcr
% 
% savestring = 'cifar-100-keras-ycbcr-fitDistr.mat';
% 
% save(savestring,'res_t');  
% % 
% 
%%
% % % % % 
% cd(asnn_root)
% addpath(genpath('./'));
% 
% for itheta = 0.03
%     
%     [snet, sinfo_rl] = snn_cifar_100(...
%         'expDir', 'data/cifar-100-keras-valid', ...
%         'modelType', 'keras_valid', 'testOnNetwork', 'snn',...
%         'thresholdvalue', itheta);
%     
%     fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
%     
%     fprintf('Mean firing rate %f Hz \n', snet.firingrate)
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
%     cd data/cifar-100-keras-valid
%     
%     savestring = ['cifar-100-keras-valid-1s-SUMSPIKES-'  num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
%         '_fr_' num2str(round(snet.firingrate)) '.mat'];
%     
%     save(savestring, 'ssnet', 'sinfo_rl','-v7.3');
%     cd(asnn_root)
%     
%     clear snet
%     clear ssnet
%     clear sinfo_rl
% end
% % 
% exit

%% arousal
% % 
cd(asnn_root)
addpath(genpath('./'));

for s_ara = 1

[snet, sinfo_rl] = snn_cifar_100(...
    'expDir', 'data/cifar-100-keras-valid', ...
    'modelType', 'keras_valid', 'testOnNetwork', 'snn',...
    'thresholdvalue', 0.1, 'arousal', s_ara);
fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);

fprintf('Mean firing rate %f Hz \n', snet.firingrate)


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


fprintf('Selectivity %f \n', sum(ssnet.ara<0.1)/numel(ssnet.ara)*100)

cd data/cifar-100-keras-valid
savestring = ['cifar-100-keras-SUMSPIKES-1.0s-arousal-valid-0.1-' num2str(s_ara) 'x' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];

save(savestring, 'ssnet', 'sinfo_rl');
cd(asnn_root)
end

exit

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




%% FIRING RATE MT
% load(fullfile(asnn_root, 'data', 'cifar-100-keras-valid', 'cifar-100-keras-valid-1s-SUMSPIKES-0.03-63.4_fr_95.mat'))
% matching time MNIST 350

% sum_spikes_im_MT=sum(ssnet.sumSpikesTime(1:350))/8000;
% MT_firing_rate = (sum_spikes_im_MT/tot_neurons)/0.350;

% load(fullfile(asnn_root, 'data', 'cifar-100-keras-valid', 'cifar-100-keras-SUMSPIKES-1.0s-arousal-valid-0.1-1x63.3875_fr_61.mat'))
% matching time MNIST AROUSAL 441
% sum_spikes_im_MT_A=sum(ssnet.sumSpikesTime(1:441))/8000;
% MT_firing_rate_A = (sum_spikes_im_MT/tot_neurons)/0.441;


%% Fan out

sum_spikes = 0;
tot_neurons = 0;
net = cnn_cifar_100_init_keras_valid
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
sum_spikes_im_MT =sum(ssnet.sumSpikesTime(1:686))/8000;
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
%% Analysis
% 
% pred = info_rl.val.pred(1,:);
% lab = info_rl.val.lab(1,:);
% er = lab == pred;
% erim = find(er == 0);
% hiim = find(ssnet.ara < 0.9);
% av = zeros(size(erim));
% for i = 1:numel(erim)
%     
%     a = find(hiim==erim(i));    
%     if isempty(a)==0
%         av(i) = a;
%     end
% end
% 
% display(sum(av>0)/numel(erim))

%% load high frequency res
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcrth-0.03-64.69_fr_95.mat')
spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
testHfr = ssnet.testPreSML;

%% load low frequency res
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcr-Test53.67_fr_13.mat')
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-ycbcr-Test-52.96_fr_13.mat')


spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim_l = find(ser == 0);
testLfr = ssnet.testPreSML;



% % shiim = find(ssnet.ara < 0.9);
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
cc = zeros(10000,700);
dd = zeros(10000,1);

xti = 390:10:700 ;
yti = [];
sara = [];
for i=350:700
   
%       aa = testLfr(:,i);
      bb(:,i) = bb(:,i-1) + testLfr(:,i);
      aa = reshape(bb(:,i),5,10000);
      aa = sort(aa,'descend');
      
      % Method 1 - cumulated divergence 1st vs 2nd
      cc(:,i) = aa(1,:)-aa(2,:);     
      
      
      
      if any(i == xti)          
          eth = max(cc(corrToWrong,i));
          yti = [yti, round(eth,-1)];
%           ara(cc(:,i)<ceil(eth))=1;
sara = [sara, sum(cc(:,i)<round(eth,-1))];
      end     
      
      
end

% figure
plot(xti,yti)
hold on

%
p = polyfit(xti,yti+20,1);

plot(xti,xti*p(1)+p(2))

%%
% % load arousal res
% load('data/cifar-100-keras-ycbcr/cifar-100-keras-arousal-0.25-5x63.16_fr_35.mat')
% arousal = ssnet.ara;
% 
% highl = find(arousal<0.2);
% 
% falsePos = [];
% fasleNeg = [];
% truePos = [];
% 
% for i = 1:numel(corrToWrong)
%    
%     a = find(highl==corrToWrong(i));
%     if isempty(a)
%         fasleNeg = [fasleNeg, corrToWrong(i)];
%     else
%         truePos = [truePos, corrToWrong(i)];
%     end
%     
% end
% 
% falsePos = numel(highl) - numel(truePos);
% 
% disp(['highlithed ' num2str(numel(highl))]);
% disp(['to highlith ' num2str(numel(corrToWrong))]);
% disp(['correctly highlithed ' num2str(100*numel(truePos)/numel(corrToWrong)) '%']);
% 
% %% some plots
% 
% corTW = [];
% 
% for i = 1:numel(corrToWrong)
%    
%     corTW = [corTW, 10*corrToWrong(i)-9:corrToWrong(i)*10]; 
%     
%     
% end
% 
% corToWH = testHfr(corTW,:);
% corToWL = testLfr(corTW,:);
% 
% %%
% figure
% subplot(1,2,1), plot(corToWH(21:30,:)')
% subplot(1,2,2), plot(corToWL(21:30,:)')
% 
% %%
% 
% ctwnorm = (corToWL(21:30,:)-mean(corToWL(21:30,:),1))./std(corToWL(21:30,:),[],1);
% cnorm = (testLfr(11:20,:)-mean(testLfr(1:10,:),1))./std(testLfr(11:20,:),[],1);
% % 
% % figure
% % subplot(1,2,1), plot(testLfr(1:10,:)')
% % subplot(1,2,2), plot(corToWL(21:30,:)')
% 
% figure
% subplot(1,2,1), plot(ctwnorm')
% subplot(1,2,2), plot(cnorm')
% 
% 
% %%
% ara = zeros(1,10000);
% ara2 = zeros(1,10000);
% for i=200:25:500
%    
%       aa = testLfr(:,i);
%       aa = reshape(aa,10,10000);
%       aa = sort(aa,'descend');
%       
%       %       bb = (aa - mean(aa,1))./std(aa,[],1);
%       %       bb = aa(2,:)./aa(1,:);
%       bbs = std(aa,[],1);
%       bbm = mean(aa,1);
%       % keep previous positives
%       ara(aa(1,:)-aa(2,:) < 5*0.2938)=1;
%       ara2(aa(1,:)-aa(2,:) < 1*bbs)=1;
%       
% end
% 
% sum(ara)
% sum(ara2)
% 
% %%
% 
% ara = zeros(1,625);
% bb = zeros(size(testLfr));
% bb(:,1) = testLfr(:,1);
% cc = zeros(625,500);
% dd = zeros(625,1);
% % ara2 = zeros(1,10000);
% for i=250:500
%    
% %       aa = testLfr(:,i);
%       bb(:,i) = bb(:,i-1) + testLfr(:,i);
%       aa = reshape(bb(:,i),100,625);
%       aa = sort(aa,'descend');
%       
%       
%       cc(:,i) = aa(1,:)-aa(2,:);
%       
%       dd = cc(:,i) - cc(:,i-1);
%       if i == 375
%           %           ara(dd<1)=1;
%           ara(cc(:,i)<890)=1;
%       end
%       
%       
%       % 300 - 300
%       % 325 - 480
%       % 350 - 680
%       % 375 - 890
%       
%       %       bb = (aa - mean(aa,1))./std(aa,[],1);
%       %       bb = aa(2,:)./aa(1,:);
% %       bbs = std(aa,[],1);
% %       bbm = mean(aa,1);
%       % keep previous positives
% %       ara(aa(1,:)-aa(2,:) < 5*0.2938)=1;
% %       ara(aa(1,:) < 0)=1;
% %       ara2(aa(1,:)-aa(2,:) < 1*bbs)=1;
%       
% end
% 
% sum(ara)
% % sum(ara2)
% 
% %
% 
% highl = find(ara==1);
% falsePos = [];
% fasleNeg = [];
% truePos = [];
% 
% for i = 1:numel(corrToWrong)
%    
%     a = find(highl==corrToWrong(i));
%     if isempty(a)
%         fasleNeg = [fasleNeg, corrToWrong(i)];
%     else
%         truePos = [truePos, corrToWrong(i)];
%     end
%     
% end
% 
% % numel(truePos)
% 
% disp(['highlithed ' num2str(numel(highl))]);
% disp(['to highlith ' num2str(numel(corrToWrong))]);
% disp(['correctly highlithed ' num2str(100*numel(truePos)/numel(corrToWrong)) '%']);

%% Compute threshold-line training

ara = zeros(1,50000);
ara2 = zeros(1,50000);
bb = zeros(size(testLfr));
bb(:,1) = testLfr(:,1);
cc = zeros(50000,500);
dd = zeros(50000,1);

xti = 390:10:410 ;
yti = [];
sara = [];
for i=350:410
   
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
sara = [sara, sum(cc(:,i)<round(eth,-1))];
      end     
      
      
end

% figure
plot(xti,yti)
hold on

%
p = polyfit(xti,yti+20,1);

plot(xti,xti*p(1)+p(2))
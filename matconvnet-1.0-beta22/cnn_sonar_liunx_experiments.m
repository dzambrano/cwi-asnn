%% Experiment with the cnn_cifar_fc_bnorm
% clc
clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn

 

rng(110)

% 
%  vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', 'local/cudnn-rc51') ;
% 

% vl_compilenn
%% importing keras
% 

[net_rl, info_rl] = snn_sonar(...
  'expDir', 'data/sonar','modelType', 'keras', 'testOnNetwork', 'cnn');
fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
% 
% % savestring = 'cifar-keras.mat';
% % save(savestring,'info_rl');
% %% 
% % res_t = fixedThresholdLayersKeras(info_rl.res,net_rl);
% % 
% % cd data/sonar
% % 
% % savestring = 'sonar-fitDistr.mat';
% % 
% % save(savestring,'res_t');  
% % 
% % 
% %%
% % % % 
% cd(asnn_root)
% addpath(genpath('./'));
% 
% [snet, sinfo_rl] = snn_sonar(...
%   'expDir', 'data/sonar', 'modelType', 'keras', 'testOnNetwork', 'snn');
% fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
% 
% fprintf('Mean firing rate %f Hz \n', snet.firingrate)
% 
% cd data/sonar
% savestring = ['sonar-' num2str((1-sinfo_rl.val(end).top1err)*100)...
%     '_fr_' num2str(round(snet.firingrate)) '.mat'];
%  
% save(savestring, 'snet', 'sinfo_rl');
% cd(asnn_root)
% 
% 

%% Inference
% 

cd(asnn_root)
addpath(genpath('./'));
% for itheta = 0.2 : 0.05: 0.8
for itheta = 0.32
    [snet, sinfo_rl] = snn_sonar(...
        'expDir', 'data/sonar', 'modelType', ...
        'keras', 'testOnNetwork', 'snn', ...
        'thresholdvalue', itheta);
    fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
    
    fprintf('Mean firing rate %f Hz \n', snet.firingrate)
    
    
    ssnet = {};
    ssnet.layers = {};
    ssnet.best = snet.best;
    ssnet.meta = snet.meta;
    ssnet.meanfiring = snet.meanfiring;
    ssnet.firingrate = snet.firingrate;
    ssnet.testPreSML = snet.testPreSML;

    
    for i=1:numel(snet.layers)
        
        if strcmp(snet.layers{i}.type,'snn_relu_th')
            
            ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
                'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
                'mean_spike', snet.layers{i}.mean_spike) ;
        else
            ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;
            
        end
    end
    
    
    
    cd data/sonar
    savestring = ['sonar-bnth-training-' num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];
    
    display(savestring)
%     save(savestring, 'ssnet', 'sinfo_rl');
    cd(asnn_root)
    
    clear snet
    clear ssnet
    clear sinfo_rl
    
end


% 
% %% ARAUSAL
% % 
% cd(asnn_root)
% addpath(genpath('./'));
% 
% for s_ara = 0.35
% 
% [snet, sinfo_rl] = snn_sonar(...
%     'expDir', 'data/sonar', 'modelType', ...
%         'keras', 'testOnNetwork', 'snn', ...
%         'thresholdvalue', s_ara, 'arausal', 1);
% fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
% 
% fprintf('Mean firing rate %f Hz \n', snet.firingrate)
% 
% 
% ssnet = {};
% ssnet.layers = {};
% ssnet.best = snet.best;
% ssnet.meta = snet.meta;
% ssnet.meanfiring = snet.meanfiring;
% ssnet.firingrate = snet.firingrate;
% ssnet.ara = snet.ara;
% 
% for i=1:numel(snet.layers)
%     
%     if strcmp(snet.layers{i}.type,'snn_relu_th')
%         
%         ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
%             'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
%             'mean_spike', snet.layers{i}.mean_spike) ;
%     else
%         ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;
%         
%     end
% end
% 
% 
% fprintf('Selectivity %f \n', sum(ssnet.ara<0.3)/numel(ssnet.ara)*100)
% 
% cd data/sonar
% savestring = ['sonar-ARAUSAL-' num2str(s_ara) '-0.15-50ms-' num2str((1-sinfo_rl.val(end).top1err)*100)...
%     '_fr_' num2str(round(snet.firingrate)) '.mat'];
% 
% save(savestring, 'ssnet', 'sinfo_rl');
% cd(asnn_root)
% end


% 
%% load high frequency res
% load('data/sonar/sonar-bnth-89.4231_fr_20.mat')
spred = sinfo_rl.val.pred(1,:);
slab = sinfo_rl.val.lab(1,:);
ser = slab == spred;
serim = find(ser == 0);
testHfr = ssnet.testPreSML;

%% load low frequency res
% load('data/sonar/sonar-bnth-81.7308_fr_9.mat') %0.9
% load('data/sonar/sonar-bnth-83.6538_fr_10.mat') %0.85
% load('data/sonar/sonar-bnth-83.6538_fr_10.mat') %0.8
% load('data/sonar/sonar-bnth-84.6154_fr_11.mat') %0.75
% load('data/sonar/sonar-bnth-84.6154_fr_12.mat') %0.7



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
ara = zeros(1,104);
ara2 = zeros(1,104);
bb = zeros(size(testLfr));
bb(:,1) = testLfr(:,1);
cc = zeros(104,500);
dd = zeros(104,1);

xti = 200:20:410 ;
yti = [];
for i=150:500
   
%       aa = testLfr(:,i);
      bb(:,i) = bb(:,i-1) + testLfr(:,i);
      aa = reshape(bb(:,i),2,104);
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

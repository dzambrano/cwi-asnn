%% Experiment with the cnn_mnist_fc_bnorm
clear all
close all

addpath(genpath('./'));
run  matlab/vl_setupnn

% this version:
% test set and validation set
% sum of spikes

%%
% compile
% vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-7.5', ...
%                'cudaMethod', 'nvcc')
%
% CURRENT
%  vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', 'local/cudnn-rc51') ;
% % DOES NOT COMPILE
%  vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', 'local/cudnn-rc6') ;

%% train network

%% Experiment with the cnn_mnist_ReLu
% This code train a ReLu-based NN (each conv or maxpool layer is followed by a ReLu layer)
% clc
% clear all
% close all
% cd(asnn_root)
% addpath(genpath('./'));
% run  matlab/vl_setupnn



rng(110)

%% importing keras
%
%
% [net_rl, info_rl] = snn_mnist(...
%   'expDir', 'data/mnist-keras-bn', 'testOnNetwork', 'cnn');
% fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
% %99.59

%% Inference
% %

cd(asnn_root)
addpath(genpath('./'));
% for itheta = 0.25 : 0.05: 0.45
for itheta = [0.12]
    [snet, sinfo_rl] = snn_mnist(...
        'expDir', 'data/mnist-keras-bn', ...
        'testOnNetwork', 'snn', ...
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
    ssnet.sumSpikesTime = snet.sumSpikesTime;

    for i=1:numel(snet.layers)

        if strcmp(snet.layers{i}.type,'snn_relu_th')

            ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
                'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
                'mean_spike', snet.layers{i}.mean_spike, ...
                'tot_spikes',snet.layers{i}.tot_spikes) ;

        else
            ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;

        end
    end



    cd data/mnist-keras-val
    savestring = ['mnist-keras-testSUMSPIKES-' num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];

    % save(savestring, 'ssnet', 'sinfo_rl');
    cd(asnn_root)
end



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
avg_firing_rate = (sum_spikes_im / tot_neurons) / 0.5;

sum_spikes_im_b = sum(ssnet.sumSpikesTime)/8000;

%% FIRING RATE MT
% load(fullfile(asnn_root, 'data', 'mnist-keras-bn', 'mnist-keras-testSUMSPIKES-0.12-99.5875_fr_49.mat'))
% matching time MNIST 350

sum_spikes_im_MT=sum(ssnet.sumSpikesTime(1:350))/8000;
MT_firing_rate = (sum_spikes_im_MT/tot_neurons)/0.350;

% load(fullfile(asnn_root, 'data', 'mnist-keras-bn', 'mnist-keras-AROUSAL-valid-test-SUMSPIKE-0.6-0.12-50ms-99.5125_fr_11.mat'))
% matching time MNIST AROUSAL 441
sum_spikes_im_MT_A=sum(ssnet.sumSpikesTime(1:441))/8000;
MT_firing_rate_A = (sum_spikes_im_MT/tot_neurons)/0.441;


%% Fan out

sum_spikes = 0;
tot_neurons = 0;
net = cnn_mnist_init_keras_BPlusC;
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
sum_spikes_im_MT =sum(ssnet.sumSpikesTime(1:441))/8000;
rats = sum_spikes_im_MT/sum_spikes_im;

spike_fo_MT = spike_fo * rats;

%% AROUSAL
% % % %
% cd(asnn_root)
% addpath(genpath('./'));
% %
% %
% for s_ara = 0.9
%
% [snet, sinfo_rl] = snn_mnist(...
%   'expDir', 'data/mnist-keras-bn', 'testOnNetwork', 'snn',...
%   'thresholdvalue', s_ara, 'arousal', 1);
% fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
%
% fprintf('Mean firing rate %f Hz \n', snet.firingrate)
%
% fprintf('Selectivity %f \n', sum(snet.ara<0.5)/numel(snet.ara)*100)
% % %
% % cd data/mnist-keras-bn
% % savestring = ['mnist-keras-ARAUSAL-4x' num2str((1-sinfo_rl.val(Q  <  end).top1err)*100)...
% %     '_fr_' num2str(round(snet.firingrate)) '.mat'];
% %
% % save(savestring, 'snet', 'sinfo_rl');
% % cd(asnn_root)
%
%
%
% %
%
%
% ssnet = {};
% ssnet.layers = {};
% ssnet.best = snet.best;
% ssnet.meta = snet.meta;
% ssnet.meanfiring = snet.meanfiring;
% ssnet.firingrate = snet.firingrate;
% ssnet.ara = snet.ara;
% ssnet.testPreSML = snet.testPreSML;
%
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
% %
% %
% %
% cd data/mnist-keras-bn
% savestring = ['mnist-keras-AROUSAL-training-' num2str(s_ara) '-0.12-50ms-' num2str((1-sinfo_rl.val(end).top1err)*100)...
%     '_fr_' num2str(round(snet.firingrate)) '.mat'];
%
% save(savestring, 'ssnet', 'sinfo_rl');
% cd(asnn_root)
% end
% %
% %% Analysis
% %
% % pred = info_rl.val.pred(1,:);
% % lab = info_rl.val.lab(1,:);
% % er = lab == pred;
% % erim = find(er == 0);
% % hiim = find(ssnet.ara < 0.9);
% % av = zeros(size(erim));
% % for i = 1:numel(erim)
% %
% %     a = find(hiim==erim(i));
% %     if isempty(a)==0
% %         av(i) = a;
% %     end
% % end
% %
% % display(sum(av>0)/numel(erim))
% %
%
% %% load high frequency res
% % load('data/mnist-keras-bn/mnist-keras-bnth0.12-99.56_fr_56.mat')
% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
% ser = slab == spred;
% serim = find(ser == 0);
% testHfr = ssnet.testPreSML;
%
% %% load low frequency res
% % load('data/mnist-keras-bn/mnist-keras-bnth0.5-99.49_fr_15.mat') % 0.5
% % load('data/mnist-keras-bn/mnist-keras-bnth0.6-99.34_fr_12.mat') % 0.6
% % load('data/mnist-keras-bn/mnist-keras-bnth0.7-99.1_fr_10.mat') % 0.7
% % load('data/mnist-keras-bn/mnist-keras-bnth0.8-98.9_fr_9.mat') % 0.8
% % load('data/mnist-keras-bn/mnist-keras-bnth0.9-98.03_fr_8.mat') % 0.9
% % load('data/mnist-keras-bn/mnist-keras-bnth1.0-96.53_fr_7.mat') %1.0
% % load('data/mnist-keras-bn/mnist-keras-bnth1.2-91.07_fr_6.mat') %1.2
% % load('data/mnist-keras-bn/mnist-keras-bnth1.4-86.13_fr_5.mat') %1.4
%
%
% spred = sinfo_rl.val.pred(1,:);
% slab = sinfo_rl.val.lab(1,:);
% ser = slab == spred;
% serim_l = find(ser == 0);
% testLfr = ssnet.testPreSML;% % shiim = find(ssnet.ara < 0.9);
% % av = zeros(size(serim));
% % for i = 1:numel(serim)
% %
% %     a = find(erim==serim(i));
% %     if isempty(a)==0
% %         av(i) = a;
% %     end
% % end
% %
% % display(sum(av>0)/numel(serim))
% %
% %
% falsePos = [];
% fasleNeg = [];
% corrToWrong = [];
%
% for i = 1:numel(serim_l)
%
%     a = find(serim==serim_l(i));
%     if isempty(a)
%        corrToWrong =  [corrToWrong, serim_l(i)];
%     end
%
% end
%
% wrongToCorr = [];
%
% for i = 1:numel(serim)
%
%     a = find(serim_l==serim(i));
%     if isempty(a)
%        wrongToCorr =  [wrongToCorr, serim(i)];
%     end
%
% end
% %%
% load(fullfile(asnn_root, 'data', 'mnist-keras-bn', 'mnist-keras-ARAUSAL-1.2-5x99.5_fr_16.mat'))
% arausal = ssnet.ara;
%
% highl = find(arausal<0.9);
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
%
% %%
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
% subplot(1,2,1), plot(corToWH(1:10,:)')
% subplot(1,2,2), plot(corToWL(1:10,:)')
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
% bb = zeros(size(testLfr));
% bb(:,1) = testLfr(:,1);
% cc = zeros(10000,500);
% dd = zeros(10000,1);
%
% xti = 350;
%
% for i=200:500
%
% %       aa = testLfr(:,i);
%       bb(:,i) = bb(:,i-1) + testLfr(:,i);
%       aa = reshape(bb(:,i),10,10000);
%       aa = sort(aa,'descend');
%
% % Method 1 - cumulated divergence 1st vs 2nd
%       cc(:,i) = aa(1,:)-aa(2,:);
%
% %       dd = cc(:,i) - cc(:,i-1);
%       if i == xti
%           %           ara(dd<1)=1;
%           ara(cc(:,i)<1350)=1;
% %           ara(cc(:,i)<(p(1)*i+p(2)))=1;
%
%       end
% % 1.0
% % th=470 4730 -> 100%  200ms to 250ms
% % th=900 3894 -> 100%  200ms to 300ms
% % th=1350 3815 -> 100%  200ms to 350ms
%
% % 1.2
% % th=440 5246 -> 100%  200ms to 250ms
% % th=890 5178 -> 100%  200ms to 300ms
% % th=1350 5205 -> 100%  200ms to 350ms
%
%
%
% % Method 2 - cumulated divergence 1st vs mean
% %       cc(:,i) = aa(1,:)-mean(aa(2:end,:));
% %
% %       dd = cc(:,i) - cc(:,i-1);
% %       if i == 350
% %           %           ara(dd<1)=1;
% %           ara(cc(:,i)<2000)=1;
% %       end
%
% % th=600 4742 -> 100%  200ms to 250ms
% % th=1200 4045 -> 100%  200ms to 300ms
% % th=1800 4504 -> 100%  200ms to 350ms
%
%
%
% %       bb = bb + aa;
% %       aa = sort(aa,'descend');
%
%       %       bb = (aa - mean(aa,1))./std(aa,[],1);
%       %       bb = aa(2,:)./aa(1,:);
%       %       bbs = std(aa,[],1);
%       %       bbm = mean(aa,1);
%
%
%       % keep previous positives
% %       ara(aa(1,:)-aa(2,:) < 5*1.1841)=1;
% %       ara2(aa(1,:)-aa(2,:) < 2*bbs)=1;
%
% end
%
% % sum(ara)
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
% numel(truePos)
%
% disp(['highlithed ' num2str(numel(highl))]);
% disp(['to highlith ' num2str(numel(corrToWrong))]);
% disp(['correctly highlithed ' num2str(100*numel(truePos)/numel(corrToWrong)) '%']);
% %
% % xhih = [xhih, xti];
% % yhih = [yhih, numel(highl)];
%
%
%
% %% Compute threshold-line
% ara = zeros(1,10000);
% ara2 = zeros(1,10000);
% bb = zeros(size(testLfr));
% bb(:,1) = testLfr(:,1);
% cc = zeros(10000,500);
% dd = zeros(10000,1);
%
% xti = 220:20:400 ;
% yti = [];
% sara = [];
% for i=200:500
%
% %       aa = testLfr(:,i);
%       bb(:,i) = bb(:,i-1) + testLfr(:,i);
%       aa = reshape(bb(:,i),10,10000);
%       aa = sort(aa,'descend');
%
%       % Method 1 - cumulated divergence 1st vs 2nd
%       cc(:,i) = aa(1,:)-aa(2,:);
%
%
%
%       if any(i == xti)
%           eth = max(cc(corrToWrong,i));
%           yti = [yti, round(eth,-1)];
%
%           sara = [sara, sum(cc(:,i)<round(eth,-1))];
% %           ara(cc(:,i)<ceil(eth))=1;
%       end
%
%
% end
%
% % figure
% plot(xti,yti)
% hold on
% %%
%
% p = polyfit(xti,yti+20,1);
%
% plot(xti,xti*p(1)+p(2))
%
%
% %%
%
% p = polyfit(xti,yti05+20,1);
%
% plot(xti,xti*p(1)/0.5459+p(2)/0.5459)
%
% %% 0.8230
% % yti05=yti;
% % yti06=yti;
% % yti07=yti;/0.8230;
% % yti08=yti;/0.8230;
% % yti09=yti;/0.8230;
% % yti10=yti;/1.0028;
% % yti12=yti;/1.1841;
% % yti14=yti;/1.3709;
%
% figure
% subplot(1,2,1), plot(xti,yti05)
% hold on
% plot(xti,yti06)
% plot(xti,yti07)
% plot(xti,yti08)
% plot(xti,yti09)
% plot(xti,yti10)
% plot(xti,yti12)
% plot(xti,yti14)
%
% subplot(1,2,2), plot(xti,yti05/0.5)
% hold on
% plot(xti,yti06/0.6)
% plot(xti,yti07/0.7)
% plot(xti,yti08/0.8)
% plot(xti,yti09/0.9)
% plot(xti,yti10/1.0)
% plot(xti,yti12/1.2)
% plot(xti,yti14/1.4)
%
% Myti = [yti05;
%     yti06;
%     yti07;
%     yti08;
%     yti09;
%     yti10;
%     yti12;
%     yti14];
%
% Mytin1 = [yti05/0.5;
%     yti06/0.6;
%     yti07/0.7;
%     yti08/0.8;
%     yti09/0.9;
%     yti10/1.0;
%     yti12/1.2;
%     yti14/1.4];
%
% Mytin2 = [yti05/0.5459;
%     yti06/0.6402;
%     yti07/0.7323;
%     yti08/0.8230;
%     yti09/0.9130;
%     yti10/1.0028;
%     yti12/1.1841;
%     yti14/1.3709];
% mti = mean(Myti,1);
% mtin1 = mean(Mytin1,1);
% mtin2 = mean(Mytin2,1);
% sti = std(Myti,[],1);
% stin1 = std(Mytin1,[],1);
% stin2 = std(Mytin2,[],1);
%
% %%
% figure
%
% subplot(1,2,1)
% h1 = shadedErrorBar(xti,mti,sti);
% hold on
% h2 = shadedErrorBar(xti,mtin1,stin1,'r');
%
% ax = gca;
% set(ax, ...
%     'XLim'        , [220 300],...
%   'YLim'        , [0 2000])
% subplot(1,2,2)
% h1 = shadedErrorBar(xti,mti,sti);
% hold on
% h2 = shadedErrorBar(xti,mtin2,stin2,'b');
%
% ax = gca;
% set(ax, ...
%     'XLim'        , [220 300],...
%   'YLim'        , [0 2000])
%
%
% % subplot(1,2,2), plot(xti,yti05/0.5459)
% % hold on
% % plot(xti,yti06/0.6402)
% % plot(xti,yti07/0.7323)
% % plot(xti,yti08/0.8230)
% % plot(xti,yti09/0.9130)
% % plot(xti,yti10/1.0028)
% % plot(xti,yti12/1.1841)
% % plot(xti,yti14/1.3709)
% %% training threshold line
%
%
% ara = zeros(1,60000);
% ara2 = zeros(1,60000);
% bb = zeros(size(testLfr));
% bb(:,1) = testLfr(:,1);
% cc = zeros(60000,500);
% dd = zeros(60000,1);
%
% xti = 220:20:400 ;
% yti = [];
% sara = [];
% for i=200:500
%
% %       aa = testLfr(:,i);
%       bb(:,i) = bb(:,i-1) + testLfr(:,i);
%       aa = reshape(bb(:,i),2,60000);
%       aa = sort(aa,'descend');
%
%       % Method 1 - cumulated divergence 1st vs 2nd
%       cc(:,i) = aa(1,:)-aa(2,:);
%
%
%
%       if any(i == xti)
%           eth = max(cc(corrToWrong,i));
%           yti = [yti, round(eth,-1)];
%
%           sara = [sara, sum(cc(:,i)<round(eth,-1))];
% %           ara(cc(:,i)<ceil(eth))=1;
%       end
%
%
% end
%
% % figure
% plot(xti,yti)
% hold on
% %%
%
% p = polyfit(xti,yti+20,1);
%
% plot(xti,xti*p(1)+p(2))

%% Experiment with the cnn_cifar_fc_bnorm
% clc
clear all
close all

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn



rng(110)
% vl_compilenn

%%
%
[net_rl, info_rl] = snn_imagenet(...
  'expDir', 'data/imagenet-keras-resnet18','modelType', 'resnet-18-keras', 'testOnNetwork', 'cnn');
fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
fprintf('Accuracy Top5 %f \n', (1-info_rl.val(end).top5err)*100);
scores=squeeze(gather(info_rl.res.x{end-1})) ;
[bestScore, best] = max(scores) ;

%% Inference
%
% %
cd(asnn_root)
% addpath(genpath('./'));
% for itheta = 0.25 : 0.05: 0.45
for iba = 3

    itheta = 0.2;
    [snet, sinfo_rl] = snn_imagenet(...
        'expDir', 'data/imagenet-keras-resnet18', 'modelType', ...
        'resnet-18-keras', 'testOnNetwork', 'snn', ...
        'thresholdvalue', itheta, 'ibatch', iba);
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



    cd data/imagenet-keras-resnet18
    savestring = ['imagenet-keras-resnet18-b' num2str(iba) '-' num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
        '_fr_' num2str(round(snet.firingrate)) '.mat'];

    save(savestring, 'ssnet', 'sinfo_rl','-v7.3');
    cd(asnn_root)


%     clear snet
%     clear ssnet
%     clear sinfo_rl

    
%     clearvars -except iba
%     g = gpuDevice(1);
%     reset(g);
end
% exit

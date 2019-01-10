function asn_imagenet(iba)
% It runs simulation of adaptive spiking neural network on ResNet18.
% The simulation runs over the test set: 50k images
% in 10 maxi-batches (iba)
% each maxi-batch runs 50 batches of 100 images each


cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn
rng(110)

cd(asnn_root)

itheta = 0.05; % set here the firing rate (theta and mf)

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



cd data/imagenet-keras-resnet18
savestring = ['imagenet-keras-valid-resnet18-SUMSPIKE-b' num2str(iba) '-' num2str(itheta) '-' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];

save(savestring, 'ssnet', 'sinfo_rl','-v7.3');
cd(asnn_root)



exit
end

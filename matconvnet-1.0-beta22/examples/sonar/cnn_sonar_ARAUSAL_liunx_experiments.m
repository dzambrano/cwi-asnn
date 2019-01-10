%% AROUSAL
% 
cd(asnn_root)
addpath(genpath('./'));

for s_ara = 1.3

[snet, sinfo_rl] = snn_sonar(...
    'expDir', 'data/sonar', 'modelType', ...
        'keras', 'testOnNetwork', 'snn', ...
        'thresholdvalue', s_ara, 'arousal', 1);
fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);

fprintf('Mean firing rate %f Hz \n', snet.firingrate)


ssnet = {};
ssnet.layers = {};
ssnet.best = snet.best;
ssnet.meta = snet.meta;
ssnet.meanfiring = snet.meanfiring;
ssnet.firingrate = snet.firingrate;
ssnet.ara = snet.ara;

for i=1:numel(snet.layers)
    
    if strcmp(snet.layers{i}.type,'snn_relu_th')
        
        ssnet.layers{end+1} = struct('type', snet.layers{i}.type, ...
            'mean_spiking_im', snet.layers{i}.mean_spiking_im,...
            'mean_spike', snet.layers{i}.mean_spike) ;
    else
        ssnet.layers{end+1} = struct('type', snet.layers{i}.type) ;
        
    end
end


fprintf('Selectivity %f \n', sum(ssnet.ara<0.4)/numel(ssnet.ara)*100)

cd data/sonar
savestring = ['sonar-AR0USAL-training-' num2str(s_ara) '-0.35-50ms-' num2str((1-sinfo_rl.val(end).top1err)*100)...
    '_fr_' num2str(round(snet.firingrate)) '.mat'];

save(savestring, 'ssnet', 'sinfo_rl');
cd(asnn_root)
end

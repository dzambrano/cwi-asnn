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

[net_rl, info_rl] = snn_imagenet(...
  'expDir', 'data/imagenet-keras-resnet18','modelType', 'resnet-18-keras', 'testOnNetwork', 'cnn');
fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
fprintf('Accuracy Top5 %f \n', (1-info_rl.val(end).top5err)*100);
% scores=squeeze(gather(info_rl.res.x{end-1})) ;
% [bestScore, best] = max(scores) ;



%% image from file


net = cnn_imagenet_init_keras('networkType', 'simplenn') ;

net.layers{end} = struct('type', 'softmax') ;
% Obtain and preprocess an image.
im = imread('peppers.png') ;
im = imresize(im, net.meta.inputSize(1:2)) ;

im_ = single(im)/255 ; % note: 255 range
% Run the CNN.
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;

% display(net.meta.classes.description{best})

%% image from h5

hdf5Path = fullfile(asnn_root, 'data');
hdf5file = 'testvalid';
% cd to data directory
cd(hdf5Path)

fitDfile = dir(fullfile(hdf5file));
%info = h5info(fitDfile.name);
% try to import already cropped
batch=[3];

images = zeros(224, 224, 3, size(batch,2),'single');
labels = zeros(1, size(batch,2),'single');

for i = 1:size(batch,2)
    
    indx = batch(i);
    im_1 = h5read(fitDfile.name,'/encoded_images',[1 1 1 indx],[1 256 256 1]);
    im_2 = h5read(fitDfile.name,'/encoded_images',[2 1 1 indx],[1 256 256 1]);
    im_3 = h5read(fitDfile.name,'/encoded_images',[3 1 1 indx],[1 256 256 1]);
    
    im = cat(3,squeeze(im_1)',squeeze(im_2)',squeeze(im_3)');
    images(:,:,:,i) = single(im(17:240,17:240,:))./255;
    
    labels(i) = single(h5read(fitDfile.name,'/targets',[1 indx],[1 1])+1);

end
% return to working direcotry
cd(vl_rootnn)



net = cnn_imagenet_init_keras('networkType', 'simplenn') ;

net.layers{end} = struct('type', 'softmax') ;
res = vl_simplenn(net, images) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;

%% for pre-trained imagenet nets
% Obtain and preprocess an image.
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% im_ = im_ - net.meta.normalization.averageImage ;

im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1) ;
im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2) ;
im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3) ;

% Run the CNN.
res = vl_simplenn(net, im_) ;

% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;

display(net.meta.classes.description{best})

%% 
asnet = net;


if strcmp(net.layers{1}.type,'conv')
    
    % add an input spiking layer if ReLU
    firstlayer = cell(1,1); %struct('type', 'snn_relu'));
    firstlayer{1} = struct('type', 'asn', 'name', 'asn1', 'precious', 1==2, 'leak', 0) ;
    asnet.layers = [firstlayer, asnet.layers] ;
    
    
end


for i=1:numel(asnet.layers)
    
    if strcmp(asnet.layers{i}.type,'relu')
       
        asnet.layers{i}.type = 'asn';
        
    end
end

resasn = vl_simplenn(asnet, im_) ;

% Show the classification result.
scores_asn = squeeze(gather(resasn(end).x)) ;
[bestScore_a, best_a] = max(scores_asn) ;

display(net.meta.classes.description{best_a})

%%


res_t = fixedThresholdLayersKeras(info_rl.res,net_rl);

cd data/imagenet-keras-resnet18

savestring = 'imagenet-fitDistr.mat';

save(savestring,'res_t');  
% 
cd(asnn_root)


%%

cd(asnn_root)
addpath(genpath('./'));

run  matlab/vl_setupnn

[snet, sinfo_rl] = snn_imagenet(...
  'expDir', 'data/imagenet-keras-resnet18', ...
  'modelType', 'resnet-18-keras', ...
  'testOnNetwork', 'snn', ...
  'thresholdvalue', 0.1);

fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top1err)*100);
fprintf('Accuracy %f \n', (1-sinfo_rl.val(end).top5err)*100);    
fprintf('Mean firing rate %f Hz \n', snet.firingrate)


%% spiking imagenet at least 10x
% Spiking Network parameters

load('imagenet-fitDistr.mat')

for in = 1:numel(res_t)
    
    if ~isempty(res_t(in))
        res_t(in).fitWDistA = single(res_t(in).fitWDistA);
        res_t(in).fitWDistB = single(res_t(in).fitWDistB);
    end
end



t_opts.t_ref        = 0.000;
t_opts.threshold    = 10e-4;
t_opts.multip_nob   = 1;
t_opts.multip_nob_o = 10e-4*0.1;
t_opts.dt           = 0.001;
t_opts.duration     = 0.5;
t_opts.thresLayer   = struct();
t_opts.thresLayer  = res_t;


opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;

im_x = gpuArray(single(repmat(im_,1,1,1,15)));

res_snet = [];

opts.gpus = [1];
% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  asnet = vl_simplenn_move(asnet, 'gpu') ;
end

[res_snet, snet] = snn_simplenn_bnasn(asnet, im_x, [], res_snet, t_opts, ...
                      'accumulate', false,...
                      'mode', 'test', ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

scores_snet = squeeze(gather(res_snet(end).x)) ;
[bestScore_s, best_s] = max(scores_snet) ;

% display(net.meta.classes.description{best_s})
% 
% [net_rl, info_rl] = snn_cifar(...
%   'expDir', 'data/cifar-keras-ycbcr','modelType', 'keras', 'testOnNetwork', 'cnn');
% fprintf('Accuracy %f \n', (1-info_rl.val(end).top1err)*100);
% 
% %89.90 matConvNet (89.95 keras)
% 
% % savestring = 'cifar-keras.mat';
% % save(savestring,'info_rl');
% %%
% % res_t = fixedThresholdLayersKeras(info_rl.res,net_rl);
% % 
% % cd data/cifar-keras-ycbcr
% % 
% % savestring = 'cifar-keras-ycbcr-fitDistr.mat';
% % 
% % save(savestring,'res_t');  
% % 
% %%
% % 
% cd(asnn_root)
% addpath(genpath('./'));
% 
% [snet, sinfo_rl] = snn_cifar(...
%   'expDir', 'data/cifar-keras-ycbcr', 'modelType', 'keras', 'testOnNetwork', 'snn');
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
% 
% cd data/cifar-keras-ycbcr
% savestring = ['cifar-keras-ycbcr' num2str((1-sinfo_rl.val(end).top1err)*100)...
%     '_fr_' num2str(round(snet.firingrate)) '.mat'];
%  
% save(savestring, 'ssnet', 'sinfo_rl');
% cd(asnn_root)



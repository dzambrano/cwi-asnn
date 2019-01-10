%% Experiment with the cnn_mnist_fc_bnorm

cd(asnn_root)
addpath(genpath('./'));
run  matlab/vl_setupnn
%%
% just in case compile
% vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-7.5', ...
%                'cudaMethod', 'nvcc')
% 
%  vl_compilenn('enableGpu', true, ...
%                'cudaRoot', '/Developer/NVIDIA/CUDA-7.5', ...
%                'cudaMethod', 'nvcc', ...
%                'enableCudnn', true, ...
%                'cudnnRoot', 'local/cudnn-rc51') ;


%% Experiment with the cnn_mnist_ReLu
% This code train a ReLu-based NN (each conv or maxpool layer is followed by a ReLu layer) 
clc
clear all
close all
cd(asnn_root)
addpath(genpath('./'));
rng(110)

%% Train 1 epoch
[net_rl, info_rl] = cnn_components(...
  'expDir', 'data/components2', 'testComponent','poolmax','batchNormalization', false);


%%


[net_rl, info_rl] = snn_components(...
  'expDir', 'data/components2', 'testComponent','poolmax','testOnNetwork', 'cnn');
fprintf('Accuracy %f % \n', (1-info_rl.val(end).top1err)*100);
%% Lambda method


% cd(asnn_root)
% addpath(genpath('./'));

load('data/components2/imdb.mat');
im_ = images.data(:,:,:,find(images.set == 1)) ;
lab_ = images.labels(1,find(images.set == 1)) ;

im_f = im_(:,:,:,1:1000);
lab_f = lab_(1:1000);

load('data/components2/net-epoch-1.mat') ;
net_rl = net;


vl_simplenn_display(net_rl);

net_rl.layers{end}.type = 'softmax'; %for the last layer

res_f = vl_simplenn(net_rl, im_f) ;

scores = squeeze(gather(res_f(end).x)) ;
[bestScore, best] = max(scores) ;

acc = sum(lab_f==best)/size(best, 2) * 100;
fprintf('Accuracy: %2.2f%%.\n', acc);



% attention very long processing!!
res_t = fixedThresholdLayers(res_f,net_rl);


cd data/components2

savestring = 'mnist-conv2-fitDistr.mat';
% save(savestring,'res_t');


%% function [res, net] = snn_simplenn_mtf(net, x, dzdy, res, t_opts, varargin)

cd(asnn_root)
addpath(genpath('./'));

[snet, sinfo_rl] = snn_components(...
  'expDir', 'data/components2', 'testComponent','poolmax','testOnNetwork', 'snn');

fprintf('Mean firing rate %f Hz \n', snet.firingrate)

%% Histogram ANN-relu layers

close all
figure
irelu = 0;

index = reshape(1:12, 2, 6).';


irelu = irelu +1;
subplot(6,2,index(irelu))
h=histogram(info_rl.res.x{1,1});
[N,edges] = histcounts(info_rl.res.x{1,1});
Nmax=max(N(2:end));
ax = gca;
set(ax, ...
    'XLim'        , [0 max(info_rl.res.x{1,1}(:))],...
    'YLim'        , [0 Nmax],...
    'Box'         , 'off'     , ...
    'YGrid'       , 'off'      , ...
    'XGrid'       , 'off'      , ...
    'LineWidth'   , 2         );% 'YTick'       , 20:20:160 , ...
%'YMinorTick'  , 'on'      , ...


hXLabel = xlabel(['Activity ReLu Layer ' num2str(1-1)],'FontName','Calibri','FontSize',36,'FontWeight','Bold');
set(ax,'FontSize',28,'FontWeight','Bold');



for i = 2:size(info_rl.res.x,2)

    
    if strcmp(net_rl.layers{1,i-1}.type,'sumtanh')
        
        irelu = irelu +1;       
        subplot(6,2,index(irelu))
        h=histogram(info_rl.res.x{1,i});
        [N,edges] = histcounts(info_rl.res.x{1,i});
        Nmax=max(N(2:end));
        ax = gca;
        set(ax, ...
            'XLim'        , [0 max(info_rl.res.x{1,i}(:))],...            
            'YLim'        , [0 Nmax],...            
            'Box'         , 'off'     , ...
            'YGrid'       , 'off'      , ...
            'XGrid'       , 'off'      , ...            
            'LineWidth'   , 2         );% 'YTick'       , 20:20:160 , ...
            %'YMinorTick'  , 'on'      , ...
        
        
        hXLabel = xlabel(['Activity ReLu Layer ' num2str(i-1)],'FontName','Calibri','FontSize',36,'FontWeight','Bold');        
        set(ax,'FontSize',28,'FontWeight','Bold');



        
    end

end
%%
snet_rl_d = snet;
for i = 1:size(snet_rl_d.layers,2)

    
    if strcmp(snet_rl_d.layers{i}.type,'snn_relu_th')||strcmp(snet_rl_d.layers{i}.type,'snn_relu_th_MAX')
        
        irelu = irelu +1;       
        subplot(6,2,index(irelu))
        h=histogram(snet_rl_d.layers{i}.sum_spikes);
        [N,edges] = histcounts(snet_rl_d.layers{i}.sum_spikes);
        Nmax=max(N(2:end));
        ax = gca;
        set(ax, ...
            'XLim'        , [0 max(snet_rl_d.layers{i}.sum_spikes(:))],...            
            'YLim'        , [0 Nmax],...            
            'Box'         , 'off'     , ...
            'YGrid'       , 'off'      , ...
            'XGrid'       , 'off'      , ...            
            'LineWidth'   , 2         );% 'YTick'       , 20:20:160 , ...
            %'YMinorTick'  , 'on'      , ...
        
        
        hXLabel = xlabel(['Number of spikes ReLu Layer ' num2str(i-1)],'FontName','Calibri','FontSize',36,'FontWeight','Bold');        
        set(ax,'FontSize',28,'FontWeight','Bold');



        
    end

end

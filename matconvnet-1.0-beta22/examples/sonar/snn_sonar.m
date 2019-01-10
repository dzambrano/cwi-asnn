function [net, info] = snn_sonar(varargin)
% SNN_CIFAR   Spiking NeuralNetwork with MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.



run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.testOnNetwork = 'cnn';
opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('sonar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data','sonar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts.thresholdvalue = 0.1;
opts.arousal = false;


opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------


switch opts.modelType
  
    case 'keras'
        net = cnn_sonar_init_keras('networkType', opts.networkType) ;
        
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end


% if exist(opts.imdbPath, 'file')
%   imdb = load(opts.imdbPath) ;
% else
  imdb = getIrisImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
% end

net.meta.classes.name = imdb.meta.classes(:)' ;

switch opts.testOnNetwork
    case 'cnn', opts.networkType = 'cnn';
    case 'snn', opts.networkType = 'snn';
    case 'lsuv', opts.networkType = 'lsuv';
end

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
    
    
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
  case 'lsuv', trainfn = @cnn_init_lsuv ;
  case 'cnn', trainfn = @cnn_test_s ;
  case 'snn', trainfn = @snn_test ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3), ...
    'thresholdvalue', opts.thresholdvalue,...
    'arousal', opts.arousal) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'snn'
    fn = @(x,y) getSimpleSNNBatch(x,y) ;
  case 'cnn'
    fn = @(x,y) getSimpleSNNBatch(x,y) ;
  case 'lsuv'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;


% -------------------------------------------------------------------------
function [images, labels] = getSimpleSNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
% if rand > 0.5, images=fliplr(images) ; end


% -------------------------------------------------------------------------
function imdb = getIrisImdb(opts)
% -------------------------------------------------------------------------

cd ('/Volumes/Data/python/data')
% cd ('../../../python/data')
X_train = load('sonar_X_train.mat');
X_test = load('sonar_X_test.mat');
y_train = load('sonar_Y_train.mat');
y_test = load('sonar_Y_test.mat');

% cd('../../matlab/SNN/matconvnet-1.0-beta22')
cd(asnn_root)


set = [ones(1,numel(y_train.y_train)) 3*ones(1,numel(y_test.y_test))];

data = single(reshape([X_train.X_train'  X_test.X_test'],1,1,60,208));

labelsf = load('sonardata.mat');
imdb.images.data = data ;
imdb.images.labels = single(cat(2, y_train.y_train+1, y_test.y_test+1)) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = labelsf.R;

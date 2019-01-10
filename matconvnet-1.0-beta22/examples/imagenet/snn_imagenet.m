function [net, info] = snn_imagenet(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.testOnNetwork = 'cnn';


opts.dataDir = fullfile(vl_rootnn, 'data','ILSVRC2012') ;
opts.modelType = 'resnet-18-keras' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.thresholdvalue = 0.1;
opts.arousal = false;
opts.ibatch = 1;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.vl_rootnn = vl_rootnn;
opts.hdf5Path = '/Volumes/SharedHdd1/imagenet_1000cat';
opts.hdf5file = 'ilsvrc2012_valid_nc.hdf5';
% opts.hdf5file = 'ilsvrc2012_train_cut.hdf5';
% opts.imdbPath = fullfile(asnn_root, 'data', 'ilsvrc2012_valid_par.hdf5');

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------



if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = createFakeImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
                                     'colorDeviation', rgbDeviation, ...
                                     'classNames', imdb.classes.name, ...
                                     'classDescriptions', imdb.classes.description) ;
      opts.networkType = 'dagnn' ;

     case  'resnet-18-keras'
      net = cnn_imagenet_init_keras('networkType', opts.networkType) ;
    otherwise
      net = cnn_imagenet_init('model', opts.modelType, ...
                              'batchNormalization', opts.batchNormalization, ...
                              'weightInitMethod', opts.weightInitMethod, ...
                              'networkType', opts.networkType, ...
                              'averageImage', rgbMean, ...
                              'colorDeviation', rgbDeviation, ...
                              'classNames', imdb.classes.name, ...
                              'classDescriptions', imdb.classes.description) ;
  end
else
  net = opts.network ;
  opts.network = [] ;
end


switch opts.testOnNetwork
    case 'cnn', opts.networkType = 'cnn';
    case 'snn', opts.networkType = 'snn';
end



% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
  case 'cnn', trainFn = @cnn_test_s ;
  case 'snn', trainFn = @snn_test_ba ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train,...
                      'thresholdvalue', opts.thresholdvalue,...
                      'arousal', opts.arousal,...
                      'ibatch', opts.ibatch) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
% check if it has any sense
% net = cnn_imagenet_deploy(net) ;
% modelPath = fullfile(opts.expDir, 'net-deployed.mat')
% 
% switch opts.networkType
%   case 'simplenn'
%     save(modelPath, '-struct', 'net') ;
%   case 'dagnn'
%     net_ = net.saveobj() ;
%     save(modelPath, '-struct', 'net_') ;
%     clear net_ ;
% end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts)
% -------------------------------------------------------------------------
% no aigmentation
% if numel(meta.normalization.averageImage) == 3
%   mu = double(meta.normalization.averageImage(:)) ;
% else
%   mu = imresize(single(meta.normalization.averageImage), ...
%                 meta.normalization.imageSize(1:2)) ;
% end
% 
% useGpu = numel(opts.train.gpus) > 0 ;
% 
% bopts.test = struct(...
%   'useGpu', useGpu, ...
%   'numThreads', opts.numFetchThreads, ...
%   'imageSize',  meta.normalization.imageSize(1:2), ...
%   'cropSize', meta.normalization.cropSize, ...
%   'subtractAverage', mu) ;
% 
% % Copy the parameters for data augmentation
% bopts.train = bopts.test ;
% for f = fieldnames(meta.augmentation)'
%   f = char(f) ;
%   bopts.train.(f) = meta.augmentation.(f) ;
% end

fn = @(x, y) getSimpleSNNBatch(x, y) ;



% -------------------------------------------------------------------------
function [images, labels] = getSimpleSNNBatch(imdb, batch)
% -------------------------------------------------------------------------

hdf5Path = '/Volumes/SharedHdd1/imagenet_1000cat';
hdf5file = 'ilsvrc2012_valid_nc.hdf5';
% hdf5file = 'ilsvrc2012_train_cut.hdf5';
% cd to data directory
cd(hdf5Path)

fitDfile = dir(fullfile(hdf5file));
%info = h5info(fitDfile.name);
% try to import already cropped
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





% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = imdb.images.label(batch) ;
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end



% -------------------------------------------------------------------------
function imdb = createFakeImdb(opts)
% -------------------------------------------------------------------------



% cd to data directory
cd(opts.hdf5Path)

fitDfile = dir(fullfile(opts.hdf5file));



labels = h5read(fitDfile.name,'/targets',[1 1],[1 50000]);
% return to working direcotry
cd(opts.vl_rootnn)



data = cell(1, 2);

set = 2*ones(1,size(labels,2));

imdb.images.data = data ;
imdb.images.labels = single(labels) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;






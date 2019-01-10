function [net, info] = snn_cifar_100(varargin)
% SNN_CIFAR   Spiking NeuralNetwork with MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.



run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.testOnNetwork = 'cnn';
opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('cifar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts.thresholdvalue = 0.1;
opts.arausal = false;


opts = vl_argparse(opts, varargin) ;



if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------


switch opts.modelType
    case 'stanh'
        net = cnn_cifar_init_stanh_lsuv('networkType', opts.networkType) ;
        vl_simplenn_display(net);
    case 'lenet'
        net = cnn_cifar_init('networkType', opts.networkType) ;
    case 'nin'
        net = cnn_cifar_init_nin('networkType', opts.networkType) ;
    case 'keras'
        net = cnn_cifar_100_init_keras('networkType', opts.networkType) ;
        
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end


if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

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
  'val', find(imdb.images.set == 3),...
  'thresholdvalue', opts.thresholdvalue,...
  'arausal', opts.arausal) ;

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
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-100-matlab');
files = [{'train.mat'}  {'test.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([1, 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.fine_labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});

% data = single(cat(4, data{:}));


data = cat(4, data{:});
for i = 1: size(data,4)
    %     data(:,:,:,i) = rgb2ycbcr(data(:,:,:,i));
    
    rImage = data(:,:,1,i);
    gImage = data(:,:,2,i);
    bImage = data(:,:,3,i);
    
    Y =       + 0.299    * rImage + 0.587    * gImage + 0.114    * bImage;
    Cb = 128   - 0.168736 * rImage - 0.331264 * gImage + 0.5      * bImage;
    Cr = 128   + 0.5      * rImage - 0.418688 * gImage - 0.081312 * bImage;
    
    data(:,:,:,i) = cat(3, Y,Cb,Cr);
   
end
data = single(data);

% no normalization from here
% ---------
% 
% % remove mean in any case
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean);% size(data)
% 
% 
% % normalize by image mean and std as suggested in `An Analysis of
% % Single-Layer Networks in Unsupervised Feature Learning` Adam
% % Coates, Honglak Lee, Andrew Y. Ng
% 
% 
% if opts.contrastNormalization
%   z = reshape(data,[],60000) ;
%   z = bsxfun(@minus, z, mean(z,1)) ;
%   n = std(z,0,1) ;
%   z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
%   data = reshape(z, 32, 32, 3, []) ;
% end
% 
% if opts.whitenData
%   z = reshape(data,[],60000) ;
%   W = z(:,set == 1)*z(:,set == 1)'/60000 ;
%   [V,D] = eig(W) ;
%   % the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
%   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
%   data = reshape(z, 32, 32, 3, []) ;
% end
% 
% 
% data_tr = data(:,:,:,1:50000);
% data_te = data(:,:,:,50001:end);
% 
% set_tr = repmat(set(1:50000),1,5);
% 
% labels_tr = repmat(labels(1:5),1,5);
% 
% labels = [labels_tr labels(6)];
% set = [set_tr set(50001:end)];
% 
% data_tr = repmat(data_tr,1,1,1,5);
% data_tr = data_tr + 0.01*randn([size(data_tr)]);
% 
% data = cat(4,data_tr,data_te);
% 
% 

% % --------- Normalize between -1 and 1 each axes
% data = images.data;
% % 
% z1 = reshape(data(:,:,1,:),[],60000);
% z2 = reshape(data(:,:,2,:),[],60000);
% z3 = reshape(data(:,:,3,:),[],60000);
% 
% range1 = max(z1(:))-min(z1(:));
% range2 = max(z2(:))-min(z2(:));
% range3 = max(z3(:))-min(z3(:));
% 
% z1 = (z1 - min(z1(:))) / range1;
% z2 = (z2 - min(z2(:))) / range2;
% z3 = (z3 - min(z3(:))) / range3;
% 
% z1 = (z1*2) -1;
% z2 = (z2*2) -1;
% z3 = (z3*2) -1;
% 
% data = reshape([z1,z2,z3],32,32,3,60000);

% dataneg = data<0;
% datapos = data>0;
% data(datapos) = bsxfun(@rdivide, data(datapos), max(data(:)));% size(data)
% data(dataneg) = bsxfun(@rdivide, data(dataneg), abs(min(data(:))));% size(data)
% 

% 
% mdata = max(A(:));
% midata = min(A(:));
% range = mdata - midata;
% B = (A - midata) / range;
% range2 = mdata - midata;
% B = (B*range2) + midata;



% 
% mdata = max(data(:));
% midata = min(data(:));
% range = mdata - midata;
% data = (data - midata) / range;
% % range2 = mdata - midata;
% range2 = mdata/30 - midata/30;
% % range2 = 2;
% data = (data*range2) + midata/30;

% dataMin = abs(min(data(:)));
% data = bsxfun(@plus, data, dataMin);% size(data)
% dataMax = max(data(:));
% data = data/dataMax;


data = data/255;


clNames = load(fullfile(unpackPath, 'meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.fine_label_names;

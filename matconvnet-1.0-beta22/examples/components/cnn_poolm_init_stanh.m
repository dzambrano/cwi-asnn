function net = cnn_poolm_init_stanh(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;


net.layers = {} ;

% Block 1  --------------------
net.layers{end+1} = struct('type', 'sumtanh') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {init_weights(5,1,64)}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sumtanh') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'sumtanh') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {init_weights(12,64,10)}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;



% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 2) ;
  net = insertBnorm(net, 8) ;
  net = insertBnorm(net, 14) ;
%   net = insertBnorm(net, 16) ;
end

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = 0.01 ;
net.meta.trainOpts.numEpochs = 1 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.gpus = [1]; 
% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {'prediction', 'label'}, 'error') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
      'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

% --------------------------------------------------------------------
function weights = init_weights(k,m,n)
% --------------------------------------------------------------------
weights{1} = randn(k,k,m,n,'single') * sqrt(2/(k*k*m)) ;
weights{2} = zeros(n,1,'single') ;
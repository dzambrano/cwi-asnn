function net = cnn_mnist_init_keras_BPlusC(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	# create model
% 	model = Sequential()
% 	model.add(BatchNormalization(input_shape=(28, 28, 1),mode=0, axis=3,epsilon=1e-05))
% 	model.add(ASNTransfer(input_shape=(28, 28, 1)))
% 	model.add(Convolution2D(32, 3, 3, input_shape=(28, 28, 1), activation='linear', border_mode='valid',dim_ordering='tf'))
% 	model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% 	model.add(ASNTransfer())
% 	model.add(MaxPooling2D(pool_size=(2, 2)))
% 	model.add(Dropout(0.2))
% 	model.add(Convolution2D(64, 3, 3, activation='linear', border_mode='valid',dim_ordering='tf'))
% 	model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% 	model.add(ASNTransfer())
% 	model.add(MaxPooling2D(pool_size=(2, 2)))
% 	model.add(Dropout(0.2))
% 	model.add(Flatten())
% 	model.add(Dense(128, activation='linear'))
% 	model.add(BatchNormalization(epsilon=1e-05))
% 	model.add(ASNTransfer())
% 	model.add(Dense(50, activation='linear'))
% 	model.add(BatchNormalization(epsilon=1e-05))
% 	model.add(ASNTransfer())
% 	model.add(Dense(num_classes, activation='softmax'))%     
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net.layers = {} ;

% netw = readweightsbyname('mnist_noth4.h5');
netw = readweightsbyname_imagenet('mnist_noth6_valid3_20.h5');
% netw = readweightsbyname_imagenet('mnist_noth4_eta100.h5');


% Block 1  --------------------
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 1)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', 1)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 2)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', 2)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 3)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;


net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', 3)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 4)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;



net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'dense', 1)}, ... %includes flatten
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 5)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'dense', 2)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', 6)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'dense', 3)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;


% reprocess net

bnorm = [];
for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv') && strcmp(net.layers{i+1}.type,'bnorm')
        bnorm = [bnorm ,i];        
        rb = net.layers{i+1}.weights{1}./net.layers{i+1}.weights{3}(:,2);
        for j = 1:numel(rb)
            net.layers{i}.weights{1}(:,:,:,j) = bsxfun(@times,net.layers{i}.weights{1}(:,:,:,j),rb(j));
        end
        net.layers{i}.weights{2} = rb .* (net.layers{i}.weights{2} - net.layers{i+1}.weights{3}(:,1))...
            + net.layers{i+1}.weights{2};
    end
end
% remove bnorm layers
for l = 1:numel(bnorm)   
    i = bnorm(l);
    if strcmp(net.layers{i}.type,'conv') && strcmp(net.layers{i+1}.type,'bnorm')
        net.layers = horzcat(net.layers(1:i), net.layers(i+2:end)) ;        
        bnorm(l:end) = bnorm(l:end) - 1; 
    end    
end


% optionally switch to batch normalization
if opts.batchNormalization
  net = insertBnorm(net, 2) ;
  net = insertBnorm(net, 8) ;
  net = insertBnorm(net, 14) ;
%   net = insertBnorm(net, 16) ;
end

% Meta parameters
net.meta.inputSize = [28 28 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 0 ;
net.meta.trainOpts.batchSize = 2000; %2000
net.meta.trainOpts.gpus = [0]; 
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
function weights = load_weights(netw, type, n)
% --------------------------------------------------------------------

switch type
    case 'bnorm'        
          
        weights{1} = single(netw.batchn(n).gamma);
        weights{2} = single(netw.batchn(n).beta);
        weights{3} = [single(netw.batchn(n).runningmean) single(sqrt(netw.batchn(n).runningstd+1.0e-05))];
%   
    case 'conv'        
        w1 = single(netw.conv(n).weights);
        sw = size(w1);
        for i = 1:sw(1)
            for j = 1:sw(2)
                a = squeeze(w1(i,j,:,:));
                weights{1}(:,:,j,i)=a';
            end
        end        
        weights{2} = single(netw.conv(n).bias);        
    
    case 'dense'    
        if n ==  1
            
            sd = size(netw.dense(n).weights);
            % get number of filters in the last conv layer
            scn = size(netw.conv(end).weights,1); 
            % assuming square filters
            scs = sqrt(sd(2)/scn); 
            
            w1 = single(reshape(netw.dense(n).weights',scn,scs,scs,sd(1)));
            
            for i = 1:size(w1,4)
                for j = 1:size(w1,1)
                    a = squeeze(w1(j,:,:,i));
                    weights{1}(:,:,j,i)=a';
                end
            end

            weights{2} = single(netw.dense(n).bias);
            
        else
            s1 = size(netw.dense(n).weights,1);
            s2 = size(netw.dense(n).weights,2);
            weights{1} = single(reshape(netw.dense(n).weights',1,1,s2,s1));
            weights{2} = single(netw.dense(n).bias);
        end
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


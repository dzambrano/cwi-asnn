function net = cnn_cifar_100_init_keras_valid(varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;


net.layers = {} ;

lr = [1 1] ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% # Create the BIG model w/ ASNTranfer function
% model = Sequential()
% model.add(BatchNormalization(input_shape=(32, 32, 3),mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer(input_shape=(32, 32, 3)))
% model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), activation='linear', border_mode='same',dim_ordering='tf'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(64, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% 
% model.add(MaxPooling2D(pool_size=(2, 2)))
% 
% model.add(Convolution2D(128, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(128, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% 
% model.add(MaxPooling2D(pool_size=(2, 2)))
% 
% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% 
% model.add(MaxPooling2D(pool_size=(2, 2)))
% 
% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.4))
% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
% model.add(ASNTransfer())
% 
% model.add(MaxPooling2D(pool_size=(2, 2)))
% 
% model.add(Flatten())
% model.add(Dropout(0.5))
% model.add(Dense(512, activation='linear', W_constraint=maxnorm(3)))
% model.add(BatchNormalization(epsilon=1e-05))
% model.add(ASNTransfer())
% model.add(Dropout(0.5))
% 
% model.add(Dense(num_classes, activation='softmax'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% model.add(BatchNormalization(input_shape=(32, 32, 3),mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer(input_shape=(32, 32, 3)))
%% model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), activation='linear', border_mode='same',dim_ordering='tf'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(64, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
    % model.add(MaxPooling2D(pool_size=(2, 2)))
%% model.add(Convolution2D(128, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(128, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
    % model.add(MaxPooling2D(pool_size=(2, 2)))
%% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(256, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
    % model.add(MaxPooling2D(pool_size=(2, 2)))
%% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Convolution2D(512, 3, 3, activation='linear', border_mode='same'))
%% model.add(BatchNormalization(mode=0, axis=3,epsilon=1e-05))
    % model.add(ASNTransfer())
    % model.add(MaxPooling2D(pool_size=(2, 2)))
%% model.add(Flatten())
%% model.add(Dense(512, activation='linear', W_constraint=maxnorm(3)))
%% model.add(BatchNormalization(epsilon=1e-05))
    % model.add(ASNTransfer())
%% model.add(Dense(num_classes, activation='softmax'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% netw = readweights100();
% netw = readweightsbyname('Cifar100_noth_64.62.h5');
netw = readweightsbyname_cifar100('Cifar100_vgg11_63.55.h5');



nbnorm = 0;
nconv = 0;

% Block 1
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm',nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;


net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Block 2
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ; 
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;
% nconv = nconv +1;
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {load_weights(netw, 'conv', nconv)}, ...
%     'stride', 1, ...
%     'pad', 1) ;
% nbnorm = nbnorm +1;
% net.layers{end+1} = struct('type', 'bnorm', ...
%     'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
%     'learningRate', [1 1 0.05], ...
%     'weightDecay', [0 0]) ;
% net.layers{end+1} = struct('type', 'asn') ;


net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Block 3
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ; 
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ; 


net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;


% Block 4
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ; 
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', 1, ...
    'pad', 1) ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
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
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ;
net.layers{end+1} = struct('type', 'asn') ;

% 
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {load_weights(netw, 'dense', 2)}, ...
%     'stride', 1, ...
%     'pad', 0) ;
% nbnorm = nbnorm +1;
% net.layers{end+1} = struct('type', 'bnorm', ...
%     'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
%     'learningRate', [1 1 0.05], ...
%     'weightDecay', [0 0]) ;
% net.layers{end+1} = struct('type', 'asn') ;

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'dense', 2)}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;


% % reprocess net

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
        
%         net.layers = horzcat(net.layers(1:i), net.layers(i+2:end)) ;
    end
end
% remove bnorm
for l = 1:numel(bnorm)
   
    i = bnorm(l);
    if strcmp(net.layers{i}.type,'conv') && strcmp(net.layers{i+1}.type,'bnorm')
        net.layers = horzcat(net.layers(1:i), net.layers(i+2:end)) ;
        
        bnorm(l:end) = bnorm(l:end) - 1; 
    end
    
    
end


% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.002, 0.01, 0.02, 0.04 * ones(1,80), 0.004 * ones(1,10), 0.0004 * ones(1,10)] ;
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.batchSize = 500 ;
net.meta.trainOpts.numEpochs = 0;%numel(net.meta.trainOpts.learningRate) ;
net.meta.trainOpts.gpus = [1];

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'error') ;
    otherwise
        assert(false) ;
end

function weights = init_weights(k,m,n)
weights{1} = randn(k,k,m,n,'single') * sqrt(2/(k*k*m)) ;
weights{2} = zeros(n,1,'single') ;




% --------------------------------------------------------------------
function weights = load_weights(netw, type, n)
% --------------------------------------------------------------------

switch type
    case 'bnorm'
        
        weights{1} = single(netw.batchn(n).gamma);
        weights{2} = single(netw.batchn(n).beta);
        weights{3} = [single(netw.batchn(n).runningmean) single(sqrt(netw.batchn(n).runningstd+1.0e-05))];
%         %
%         weights{1} = single(netw.batchn(n).gamma);
%         weights{2} = 0*single(netw.batchn(n).beta);
%         weights{3} = [0*single(netw.batchn(n).runningmean) single(sqrt(netw.batchn(n).runningstd+1.0e-05))];
        
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
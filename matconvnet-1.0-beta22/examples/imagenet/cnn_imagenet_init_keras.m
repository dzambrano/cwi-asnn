function net = cnn_imagenet_init_keras(varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;


net.layers = {} ;

lr = [1 1] ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input_1 (InputLayer)             (None, 224, 224, 3)   0                                            
% ____________________________________________________________________________________________________
% conv2d_1 (Conv2D)                (None, 112, 112, 64)  9472                                         
% ____________________________________________________________________________________________________
% batch_normalization_1 (BatchNorm (None, 112, 112, 64)  256                                          
% ____________________________________________________________________________________________________
% lambda_1 (Lambda)                (None, 112, 112, 64)  0                                            
% ____________________________________________________________________________________________________
% max_pooling2d_1 (MaxPooling2D)   (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% conv2d_2 (Conv2D)                (None, 56, 56, 64)    36928                                        
% ____________________________________________________________________________________________________
% batch_normalization_2 (BatchNorm (None, 56, 56, 64)    256                                          
% ____________________________________________________________________________________________________
% lambda_2 (Lambda)                (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% conv2d_3 (Conv2D)                (None, 56, 56, 64)    36928                                        
% ____________________________________________________________________________________________________
% merge_1 (Merge)                  (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% batch_normalization_3 (BatchNorm (None, 56, 56, 64)    256                                          
% ____________________________________________________________________________________________________
% lambda_3 (Lambda)                (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% conv2d_4 (Conv2D)                (None, 56, 56, 64)    36928                                        
% ____________________________________________________________________________________________________
% batch_normalization_4 (BatchNorm (None, 56, 56, 64)    256                                          
% ____________________________________________________________________________________________________
% lambda_4 (Lambda)                (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% conv2d_5 (Conv2D)                (None, 56, 56, 64)    36928                                        
% ____________________________________________________________________________________________________
% merge_2 (Merge)                  (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% batch_normalization_5 (BatchNorm (None, 56, 56, 64)    256                                          
% ____________________________________________________________________________________________________
% lambda_5 (Lambda)                (None, 56, 56, 64)    0                                            
% ____________________________________________________________________________________________________
% conv2d_6 (Conv2D)                (None, 28, 28, 128)   73856                                        
% ____________________________________________________________________________________________________
% batch_normalization_6 (BatchNorm (None, 28, 28, 128)   512                                          
% ____________________________________________________________________________________________________
% lambda_6 (Lambda)                (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% conv2d_8 (Conv2D)                (None, 28, 28, 128)   8320                                         
% ____________________________________________________________________________________________________
% conv2d_7 (Conv2D)                (None, 28, 28, 128)   147584                                       
% ____________________________________________________________________________________________________
% merge_3 (Merge)                  (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% batch_normalization_7 (BatchNorm (None, 28, 28, 128)   512                                          
% ____________________________________________________________________________________________________
% lambda_7 (Lambda)                (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% conv2d_9 (Conv2D)                (None, 28, 28, 128)   147584                                       
% ____________________________________________________________________________________________________
% batch_normalization_8 (BatchNorm (None, 28, 28, 128)   512                                          
% ____________________________________________________________________________________________________
% lambda_8 (Lambda)                (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% conv2d_10 (Conv2D)               (None, 28, 28, 128)   147584                                       
% ____________________________________________________________________________________________________
% merge_4 (Merge)                  (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% batch_normalization_9 (BatchNorm (None, 28, 28, 128)   512                                          
% ____________________________________________________________________________________________________
% lambda_9 (Lambda)                (None, 28, 28, 128)   0                                            
% ____________________________________________________________________________________________________
% conv2d_11 (Conv2D)               (None, 14, 14, 256)   295168                                       
% ____________________________________________________________________________________________________
% batch_normalization_10 (BatchNor (None, 14, 14, 256)   1024                                         
% ____________________________________________________________________________________________________
% lambda_10 (Lambda)               (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% conv2d_13 (Conv2D)               (None, 14, 14, 256)   33024                                        
% ____________________________________________________________________________________________________
% conv2d_12 (Conv2D)               (None, 14, 14, 256)   590080                                       
% ____________________________________________________________________________________________________
% merge_5 (Merge)                  (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% batch_normalization_11 (BatchNor (None, 14, 14, 256)   1024                                         
% ____________________________________________________________________________________________________
% lambda_11 (Lambda)               (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% conv2d_14 (Conv2D)               (None, 14, 14, 256)   590080                                       
% ____________________________________________________________________________________________________
% batch_normalization_12 (BatchNor (None, 14, 14, 256)   1024                                         
% ____________________________________________________________________________________________________
% lambda_12 (Lambda)               (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% conv2d_15 (Conv2D)               (None, 14, 14, 256)   590080                                       
% ____________________________________________________________________________________________________
% merge_6 (Merge)                  (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% batch_normalization_13 (BatchNor (None, 14, 14, 256)   1024                                         
% ____________________________________________________________________________________________________
% lambda_13 (Lambda)               (None, 14, 14, 256)   0                                            
% ____________________________________________________________________________________________________
% conv2d_16 (Conv2D)               (None, 7, 7, 512)     1180160                                      
% ____________________________________________________________________________________________________
% batch_normalization_14 (BatchNor (None, 7, 7, 512)     2048                                         
% ____________________________________________________________________________________________________
% lambda_14 (Lambda)               (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% conv2d_18 (Conv2D)               (None, 7, 7, 512)     131584                                       
% ____________________________________________________________________________________________________
% conv2d_17 (Conv2D)               (None, 7, 7, 512)     2359808                                      
% ____________________________________________________________________________________________________
% merge_7 (Merge)                  (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% batch_normalization_15 (BatchNor (None, 7, 7, 512)     2048                                         
% ____________________________________________________________________________________________________
% lambda_15 (Lambda)               (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% conv2d_19 (Conv2D)               (None, 7, 7, 512)     2359808                                      
% ____________________________________________________________________________________________________
% batch_normalization_16 (BatchNor (None, 7, 7, 512)     2048                                         
% ____________________________________________________________________________________________________
% lambda_16 (Lambda)               (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% conv2d_20 (Conv2D)               (None, 7, 7, 512)     2359808                                      
% ____________________________________________________________________________________________________
% merge_8 (Merge)                  (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% batch_normalization_17 (BatchNor (None, 7, 7, 512)     2048                                         
% ____________________________________________________________________________________________________
% lambda_17 (Lambda)               (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% batch_normalization_18 (BatchNor (None, 7, 7, 512)     2048                                         
% ____________________________________________________________________________________________________
% lambda_18 (Lambda)               (None, 7, 7, 512)     0                                            
% ____________________________________________________________________________________________________
% average_pooling2d_1 (AveragePool (None, 1, 1, 512)     0                                            
% ____________________________________________________________________________________________________
% flatten_1 (Flatten)              (None, 512)           0                                            
% ____________________________________________________________________________________________________
% dense_1 (Dense)                  (None, 1000)          513000          

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% netw = readweights();

netw = readweightsbyname_imagenet('single_gpu_model.hdf5');


nbnorm = 0;
nconv = 0;

% Block 1 %bottom right
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;

nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [2,3,2,3]) ;

nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', [2,2], ...
    'pad', [0,1,0,1]) ;

net.layers{end+1} = struct('type', 'shortcut');

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

% MERGE 1 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');

% Block 2
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


% MERGE 2 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');


% Block 3
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,1,0,1]) ;
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

% shortcut here
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv_shortcut', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,0,0,0]) ;

% MERGE 3 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');


% Block 4
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



% MERGE 4 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');

% Block 5
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,1,0,1]) ;
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
% shortcut here
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv_shortcut', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,0,0,0]) ;

% MERGE 5 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');


% Block 6
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

% MERGE 6 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');

% Block 7
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,1,0,1]) ;
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
% shortcut here
nconv = nconv +1;
net.layers{end+1} = struct('type', 'conv_shortcut', ...
    'weights', {load_weights(netw, 'conv', nconv)}, ...
    'stride', [2,2], ...
    'pad', [0,0,0,0]) ;

% MERGE 7 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');

% Block 8
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


% MERGE 8 HERE
net.layers{end+1} = struct('type', 'sum');
net.layers{end+1} = struct('type', 'shortcut');

nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;
nbnorm = nbnorm +1;
net.layers{end+1} = struct('type', 'bnorm', ...
    'weights', {load_weights(netw,'bnorm', nbnorm)}, ...
    'learningRate', [1 1 0.05], ...
    'weightDecay', [0 0]) ; 
net.layers{end+1} = struct('type', 'asn') ;


net.layers{end+1} = struct('type', 'pool', ...
    'method', 'avg', ... % check whether this is compatible
    'pool', [7 7], ...
    'stride', 1, ...
    'pad', 0) ;


net.layers{end+1} = struct('type', 'conv', ...
    'weights', {load_weights(netw, 'dense', 1)}, ... %includes flatten
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;


% 
% 
% % % reprocess net
% 
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
net.meta.inputSize = [224 224 3] ;
net.meta.trainOpts.learningRate = [0.002, 0.01, 0.02, 0.04 * ones(1,80), 0.004 * ones(1,10), 0.0004 * ones(1,10)] ;
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.batchSize = 100 ;
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
        weights{3} = [single(netw.batchn(n).runningmean) single(sqrt(netw.batchn(n).runningstd+1.0e-03))];
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
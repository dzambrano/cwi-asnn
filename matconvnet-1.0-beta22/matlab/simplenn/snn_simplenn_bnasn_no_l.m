function [res, net] = snn_simplenn_bnasn(net, x, dzdy, res, t_opts, varargin)
%SNN_SIMPLENN  Evaluate a Simple SNN spiking-neural-network.
%   This code relies on vl_simplenn.m
%   Please refer to that file for help on the main functions.
%   This code has been modified to evaluate spiking neural networks.
%   Every relu layer and the input image will be converted on snn_relu.

% more general works for ReLU or tanh (binary spikes)

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);
t_duration = 1000*t_opts.duration;


if strcmp(net.layers{1}.type,'sumtanh')|| strcmp(net.layers{1}.type,'asn')
    net.layers{1}.type = 'snn_sumtanh';
end


n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
    doder = false ;
    if opts.skipForward
        error('simplenn:skipForwardNoBackwPass', ...
            '`skipForward` valid only when backward pass is computed.');
    end
else
    doder = true ;
end

if opts.cudnn
    cudnn = {'CuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
    cudnn = {'NoCuDNN'} ;
    bnormCudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
    if opts.skipForward
        error('simplenn:skipForwardEmptyRes', ...
            'RES structure must be provided for `skipForward`.');
    end
    res = struct(...
        'x', cell(1,n+1), ...
        'dzdx', cell(1,n+1), ...
        'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), ...
        'stats', cell(1,n+1), ...
        'time', num2cell(zeros(1,n+1)), ...
        'backwardTime', num2cell(zeros(1,n+1)), ...
        'best', cell(1,1));
end

if ~opts.skipForward
    res(1).x = x ;
end
net.best = single(zeros(size(x,4),t_duration));

%% -------------------------------------------------------------------------
%                                                              Init SNN
% -------------------------------------------------------------------------
% x = im_1; size(x)

snnopt.gpuMode = gpuMode;

snnopt.theta0 = single(t_opts.threshold); %theta0 = 0.1;

snnopt.thTau1 = single(15);
snnopt.dth1 = exp(-1/snnopt.thTau1);
snnopt.etaTau = single(50);
snnopt.dEta = exp(-1/snnopt.etaTau);

tauSig = single(.005);
snnopt.a1 = exp (-0.001/tauSig);

tauSig2 = single(.05);
snnopt.a2 = exp (-0.001/tauSig2);

snnopt.thresLayer = t_opts.thresLayer;

snnopt.islast = false;
snnopt.ismax = 0;
snnopt.isbnorm = 0;
snnopt.cudnn = cudnn;
snnopt.bnormCudnn = bnormCudnn;

snnopt.spikesize = compute_scalefactor(t_opts.threshold);

% ARAUSAL
snnopt.tsumstart = 200;
snnopt.tcheck = 250;
snnopt.kara = t_opts.do_arausal;
snnopt.do_arausal = false; % start it at tcheck
snnopt.theta0_a = 0.12;
snnopt.spikesize_a = compute_scalefactor(snnopt.theta0_a);
snnopt.arausal = false(1,size(res(1).x,4));

bb = zeros(2,size(res(1).x,4));
% find last, reset
for i=1:n
    
    
    
    if strcmp(net.layers{i}.type,'sumtanh')||strcmp(net.layers{i}.type,'relu')        
        nrelulast = i;        
    end
    
    if strcmp(net.layers{i}.type,'asn')        
        net.layers{i}.type = 'sumtanh';
        nrelulast = i;        
    end
    
    if gpuMode %save initialization time for gpus
        if strcmp(net.layers{i}.type,'snn_sumtanh')||...
                strcmp(net.layers{i}.type,'snn_relu_th')||...
                strcmp(net.layers{i}.type,'snn_sumtanh_bnorm')
            
            nrelulast = i;
            if isfield(net.layers{i},{'y'})
                
                net.layers{i} = reset_snn_layer(net.layers{i});
                
            end
        end
        if strcmp(net.layers{i}.type,'snn_sumtanh_MAX')||...
                strcmp(net.layers{i}.type,'snn_relu_th_MAX')||...
                strcmp(net.layers{i}.type,'snn_sumtanh_bnorm_max')
            
            nrelulast = i;
            if isfield(net.layers{i},{'y'})
                
                net.layers{i} = reset_snn_MAX_layer(net.layers{i});
                
            end
        end
    end
    
    % combine bnorm layer with conv
    
end

%% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------
if strcmp(net.layers{1}.type, 'bnorm')
    l = net.layers{1} ;
    res(1).x = vl_nnbnorm(res(1).x, l.weights{1}, l.weights{2}, ...
        'moments', l.weights{3}, ...
        'epsilon', l.epsilon, ...
        bnormCudnn{:}) ;
end
if strcmp(net.layers{1}.type, 'conv')
    l = net.layers{1} ;
    firstConv = vl_nnconv(res(1).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    
end


net.testPreSML = zeros(size(x,4)*2,t_duration);



for t=1:t_duration
    % Create spikes from the input images t=t+1
    %   (for all images in parallel)
    
    snnopt.isfirst = 1;
    
%     l = net.layers{1} ;

    res(1).time = tic ;
    bias = 0;
    
    if strcmp(net.layers{1}.type, 'bnorm')            
        net.layers{2}.type = 'snn_sumtanh';
        res(2).x = res(1).x;        
    end
        
    if strcmp(net.layers{1}.type, 'snn_sumtanh')                  
        if gpuMode
            if snnopt.do_arausal
                [res(i+1).x, net.layers{1}] = snn_sumtanh_gpu_a(res(i).x, net.layers{1}, bias, snnopt) ;
            else
                [res(i+1).x, net.layers{1}] = snn_sumtanh_gpu(res(i).x, net.layers{1}, bias, snnopt) ;
            end
        end
%         net.layers{1} = l;
        bias = bias*0;
        snnopt.isfirst = 0;        
    end
        
    if strcmp(l.type, 'conv')
        res(2).x = firstConv;
    end
       
    res(1).time = toc(res(1).time) ;
    
    
    for i=2:n
        
        
        if opts.skipForward, break; end;
%         l = net.layers{i} ;
        
        res(i).time = tic ;
        
        
        if strcmp(net.layers{i}.type,'conv')
            bias = net.layers{i}.weights{2};
            if snnopt.gpuMode
                lweights_2 = 0*net.layers{i}.weights{2};
            else
                lweights_2 = single(zeros(size(net.layers{i}.weights{2})));
                bias = gather(bias);
            end
        end
        
        
        if strcmp(net.layers{i}.type,'conv_shortcut')
            bias = bias + net.layers{i}.weights{2};
            if snnopt.gpuMode
                lweights_2 = 0*net.layers{i}.weights{2};
            else
                lweights_2 = single(zeros(size(net.layers{i}.weights{2})));
                bias = gather(bias);
            end
        end
        
        
        if strcmp(net.layers{i}.type, 'bnorm')
            net.layers{i+1}.type = 'snn_sumtanh_bnorm';
            snnopt.isbnorm = true;
            if strcmp(net.layers{i+2}.type, 'pool') && strcmp(net.layers{i+2}.method, 'max')
                net.layers{i+1}.type = 'snn_sumtanh_bnorm_max';
            end
        end
        
        
        if strcmp(net.layers{i}.type,'sumtanh')
            net.layers{i}.type = 'snn_sumtanh';
            if strcmp(net.layers{i+1}.type, 'pool') && strcmp(net.layers{i+1}.method, 'max')
                net.layers{i}.type = 'snn_sumtanh_MAX';
            elseif strcmp(net.layers{i+1}.type, 'dropout') && ...
                    strcmp(net.layers{i+2}.type, 'pool') && ...
                    strcmp(net.layers{i+2}.method, 'max')
                net.layers{i}.type = 'snn_sumtanh_MAX';
            end
        end
        
        
        if i == nrelulast
            % the last-spikingNeuron output is analog
            snnopt.islast = true;
        end
        
        
        switch net.layers{i}.type
            case 'conv'
                res(i+1).x = vl_nnconv(res(i).x, net.layers{i}.weights{1}, lweights_2, ...
                    'pad', net.layers{i}.pad, ...
                    'stride', net.layers{i}.stride, ...
                    net.layers{i}.opts{:}, ...
                    cudnn{:}) ;
                
            case 'convt'
                res(i+1).x = vl_nnconvt(res(i).x, net.layers{i}.weights{1}, lweights_2, ...
                    'crop', net.layers{i}.crop, ...
                    'upsample', net.layers{i}.upsample, ...
                    'numGroups', net.layers{i}.numGroups, ...
                    net.layers{i}.opts{:}, ...
                    cudnn{:}) ;
                
            case 'pool'
                if ~snnopt.ismax
                    res(i+1).x = vl_nnpool(res(i).x, net.layers{i}.pool, ...
                        'pad', net.layers{i}.pad, 'stride', net.layers{i}.stride, ...
                        'method', net.layers{i}.method, ...
                        net.layers{i}.opts{:}, ...
                        cudnn{:}) ;
                else
                    res(i+1).x = res(i).x;
                    snnopt.ismax = 0;
                end
                
            case {'normalize', 'lrn'}
                res(i+1).x = vl_nnnormalize(res(i).x, net.layers{i}.param) ;
                
            case 'softmax'
                res(i+1).x = vl_nnsoftmax(res(i).x) ;
                
            case 'loss'
                res(i+1).x = vl_nnloss(res(i).x, net.layers{i}.class) ;
                
            case 'softmaxloss'
                
                if (t_opts.do_arausal) && (t >= snnopt.tsumstart) && (t < 410)
                    
                    aa = squeeze(gather(res(i).x));
                    bb = bb + aa;
                    
                    aac = sort(bb,'descend');
                    ccn = aac(1,:)-aac(2,:);
                    
                    
                    if t == snnopt.tcheck
                        % keep previous positives
                        %                         snnopt.arausal((aa(1,:)-aa(2,:)) < snnopt.kara*snnopt.spikesize)=true;
                        % reset previous positives
                        snnopt.arausal = false(1,size(res(1).x,4));
                        % general method MNIST
%                         snnopt.arausal(ccn < (2*snnopt.spikesize)*(4.9677*t -978.8808))=true;
                        
                        % mnist 1.4
                        %                         snnopt.arausal(ccn < (9.2212*t -1.7896e+03))=true;
                        % mnist 1.2
                        %                         snnopt.arausal(ccn < (8.9533*t -1.7e+03))=true;
                        %                         snnopt.arausal(ccn < (8.933*t -1.7713e+03))=true;
                        % mnist 1.0
                        %                         snnopt.arausal(ccn < (8.6130*t -1.6869e+03))=true;
                        %                         snnopt.arausal(ccn < (8.53*t -1.6534e+03))=true;
                        % mnist 0.9
                        %                         snnopt.arausal(ccn < (8.7545*t -1.7069e+03))=true;
                        % mnist 0.8
                        %                         snnopt.arausal(ccn < (5.7818*t -1.0524e+03))=true;
                        % mnist 0.7
                        %                         snnopt.arausal(ccn < (5.6182*t -1.1016e+03))=true;
                        % mnist 0.6
%                                                 snnopt.arausal(ccn < (5.0061*t -1.0019e+03))=true;
                        % mnist 0.5
                        %                         snnopt.arausal(ccn < (4.4606*t -898.7879))=true;
                        % cifar 0.3
                        %                         snnopt.arausal(ccn < (8.7202*t -2.2061e+03))=true;
                        % cifar 0.35
%                                                 snnopt.arausal(ccn < (9.4048*t -2.3801e+03))=true;
                        % cifar 0.25
                        %                         snnopt.arausal(ccn < (10.5*t -2.655e+03))=true;
                        % cifar 0.2
                        %                         snnopt.arausal(ccn < (9.2014*t -2.3013e+03))=true;
                        % cifar 0.15
%                         snnopt.arausal(ccn < (5.6964*t -1.3855e+03))=true;
                        % cifar 0.4
%                         snnopt.arausal(ccn < (8.2014*t -2.0788e+03))=true;

                        % cifar100 0.1
%                         snnopt.arausal(ccn < (6.3757*t -2.2885e+03))=true;
%                         % cifar100 0.15
%                         snnopt.arausal(ccn < (7.2758*t -2.6213e+03))=true;
                        % cifar100 0.2
%                         snnopt.arausal(ccn < (7.667*t -2.6551e+03))=true;
                        % cifar100 0.25
%                         snnopt.arausal(ccn < (8.6441*t -3.0173e+03))=true;
                        % cifar100 0.3
%                         snnopt.arausal(ccn < (9.3*t -2.788e+03))=true;


%                         sonar 1.0
%                         snnopt.arausal(ccn < (2.6545*t -382.7273))=true;


%                         iris 1.0
%                         snnopt.arausal(ccn < (2.6545*t -382.7273))=true;

%                         iris 0.9
%                         snnopt.arausal(ccn < (2.4909*t -376.3636))=true;

%                         iris 0.8
%                         snnopt.arausal(ccn < (2.3409*t -306.8182))=true;

%                         iris 0.7
%                         snnopt.arausal(ccn < (1.6318*t -195.9091))=true;

%                         iris 0.6
%                         snnopt.arausal(ccn < (1.3318*t -131.3636))=true;

%                         iris 0.5
%                         snnopt.arausal(ccn < (1.2864*t -182.2727))=true;

%                         iris 0.45
%                         snnopt.arausal(ccn < (0.8955*t -106.8182))=true;
%                         iris 0.4
%                         snnopt.arausal(ccn < (0.6818*t -52.7273))=true;

%                         iris 0.35
%                         snnopt.arausal(ccn < (1.1273*t -115.4545))=true;

%                         sonar 0.9
%                         snnopt.arausal(ccn < (4.1818*t -560.0000))=true;

%                         sonar 0.85
%                         snnopt.arausal(ccn < (2.4909*t -320.0000))=true;
%                         sonar 0.8
%                         snnopt.arausal(ccn < (2.2455*t -265.4545))=true;
%                         sonar 0.75
%                         snnopt.arausal(ccn < (2.0091*t -196.3636))=true;
%                         sonar 0.7
%                         snnopt.arausal(ccn < (1.5864*t -168.6364))=true;


                        % check again in about 25ms
                        %                         snnopt.tcheck = snnopt.tcheck + 20;
                        %                         net = rest_snn_a(net, snnopt.arausal);
                    end
                    
                    snnopt.do_arausal=true;

                end
% % % 
                if t >=250
                    aaa = squeeze(gather(res(i).x));
                    aac = sort(aaa,'descend');
                    aac = aac(1:2,:);
                    net.testPreSML(:,t) = aac(:);
                end

                res(i+1).x = vl_nnsoftmaxloss(res(i).x, net.layers{i}.class) ;
                
        

                
            case 'relu'
                if net.layers{i}.leak > 0, leak = {'leak', net.layers{i}.leak} ; else leak = {} ; end
                res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
                
            case 'snn_relu'
                [res(i+1).x, l] = snn_relu(res(i).x, l, t, snnopt) ;
                net.layers{i} = l;
                
            % CAREFUL loosing back-compatibility with snn_relu_th    
%             case 'snn_relu_th'
%                 if ~isfield(l,{'y'})
%                     lth = snnopt.thresLayer(i).fitWDistA.*snnopt.thresLayer(i).fitWDistB;
%                     snnopt.theta0 = lth/lambdaFac;
%                     snnopt.thMag = thetaFac*snnopt.theta0;
%                 end
%                 if gpuMode
%                     [res(i+1).x, l] = snn_relu_th_gpu(res(i).x, l, bias, t, snnopt) ;
%                 else
%                     
%                     [res(i+1).x, l] = snn_relu_th(res(i).x, l, bias, t, snnopt) ;
%                 end
%                 net.layers{i} = l;
%                 bias = bias*0;
%                 
%             case 'snn_relu_th_MAX'
%                 
%                 l1 = net.layers{i+1} ;
%                 if ~isfield(l,{'y'})
%                     lth = snnopt.thresLayer(i+1).fitWDistA.*snnopt.thresLayer(i+1).fitWDistB;
%                     snnopt.theta0 = lth/lambdaFac;
%                     snnopt.thMag = thetaFac*snnopt.theta0;
%                 end
%                 if gpuMode
%                     %                     [bbb, lg] = snn_relu_th_max_gpu(im_1g, lg, l1, biasg, t, snnopt) ;
%                     
%                     [res(i+1).x, l] = snn_relu_th_max_gpu(res(i).x, l, l1, bias, t, snnopt) ;
%                 else
%                     [res(i+1).x, l] = snn_relu_th_max(res(i).x, l, l1, bias, t, snnopt) ;
%                 end
%                 net.layers{i} = l;
%                 snnopt.ismax = 1;
%                 bias = bias*0;
                
            case 'snn_sumtanh'
                
                if gpuMode                    
                    if snnopt.do_arausal
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_gpu_a(res(i).x, net.layers{i}, bias, snnopt) ;
                    else                        
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_gpu(res(i).x, net.layers{i}, bias, snnopt) ;
                    end
                else                    
                    [res(i+1).x, net.layers{i}] = snn_sumtanh(res(i).x, net.layers{i}, bias, t, snnopt) ;
                end
%                 net.layers{i} = l;
                bias = bias*0;
                snnopt.isfirst = 0;
                
            case 'snn_sumtanh_MAX'
                
                if strcmp(net.layers{i+1}.type, 'dropout')
                    l1 = net.layers{i+2} ;                    
                else
                    l1 = net.layers{i+1} ;
                end
                
                
                if gpuMode    
                    if snnopt.do_arausal
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_max_gpu_a(res(i).x, net.layers{i}, l1, bias, snnopt) ;
                    else                                            
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_max_gpu(res(i).x, net.layers{i}, l1, bias, snnopt) ;
                    end
                else
                    [res(i+1).x, net.layers{i}] = snn_sumtanh_max(res(i).x, net.layers{i}, l1, bias, t, snnopt) ;
                end
%                 net.layers{i} = l;
                snnopt.ismax = 1;
                bias = bias*0; % once you use it, you clean it
                snnopt.isfirst = 0;
                
            case 'snn_sumtanh_bnorm'
                
                
                l1 = net.layers{i-1} ;               
                
                if gpuMode
                    if snnopt.do_arausal
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm_gpu_a(res(i).x, net.layers{i}, l1, bias, snnopt) ;
                    else  
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm_gpu(res(i).x, net.layers{i}, l1, bias, snnopt) ;
                    end
                else
                    % to do
                    [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm(res(i).x, net.layers{i}, l1, bias, t, snnopt) ;
                end
%                 net.layers{i} = l;
                bias = bias*0; % once you use it, you clean it
                snnopt.isfirst = 0;
                
            case 'snn_sumtanh_bnorm_max'
                
                l1 = net.layers{i-1} ; % bnorm
                l2 = net.layers{i+1} ; % max
                
                if gpuMode
                    if snnopt.do_arausal
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm_max_gpu_a(res(i).x, net.layers{i}, l1, l2, bias, snnopt) ;
                    else  
                        [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm_max_gpu(res(i).x, net.layers{i}, l1, l2, bias, snnopt) ;
                    end
                else
                    % to do
                    [res(i+1).x, net.layers{i}] = snn_sumtanh_bnorm_max(res(i).x, net.layers{i}, l1, bias, t, snnopt) ;
                end
%                 net.layers{i} = l;
                bias = bias*0; % once you use it, you clean it
                snnopt.isfirst = 0;
                snnopt.ismax = 1;
                
            case 'sigmoid'
                res(i+1).x = vl_nnsigmoid(res(i).x) ;
                
            case 'noffset'
                res(i+1).x = vl_nnnoffset(res(i).x, net.layers{i}.param) ;
                
            case 'spnorm'
                res(i+1).x = vl_nnspnorm(res(i).x, net.layers{i}.param) ;
                
            case 'dropout'
                if testMode
                    res(i+1).x = res(i).x ;
                else
                    [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', net.layers{i}.rate) ;
                end
                
            case 'bnorm'
                if snnopt.isbnorm
                    
                    
                    res(i+1).x = res(i).x;
                    snnopt.isbnorm = 0;
                    
                else
                    
                    if testMode
                        res(i+1).x = vl_nnbnorm(res(i).x, net.layers{i}.weights{1}, net.layers{i}.weights{2}, ...
                            'moments', net.layers{i}.weights{3}, ...
                            'epsilon', net.layers{i}.epsilon, ...
                            bnormCudnn{:}) ;
                    else
                        res(i+1).x = vl_nnbnorm(res(i).x, net.layers{i}.weights{1}, net.layers{i}.weights{2}, ...
                            'epsilon', net.layers{i}.epsilon, ...
                            bnormCudnn{:}) ;
                    end
                end
                
                
            case 'pdist'
                res(i+1).x = vl_nnpdist(res(i).x, net.layers{i}.class, net.layers{i}.p, ...
                    'noRoot', net.layers{i}.noRoot, ...
                    'epsilon', net.layers{i}.epsilon, ...
                    'aggregate', net.layers{i}.aggregate, ...
                    'instanceWeights', net.layers{i}.instanceWeights) ;
                
                
            case 'shortcut'
                res(i+1).x = res(i).x;
                shortcut = res(i).x;
                
            case 'sum'
                
                res(i+1).x = res(i).x + shortcut;
                
            case 'conv_shortcut'
                shortcut = vl_nnconv(shortcut, net.layers{i}.weights{1}, lweights_2, ...
                    'pad', net.layers{i}.pad, ...
                    'stride', net.layers{i}.stride, ...
                    'dilate', net.layers{i}.dilate, ...
                    net.layers{i}.opts{:}, ...
                    cudnn{:}) ;
                res(i+1).x = res(i).x;
                
            case 'custom'
                res(i+1) = l.forward(l, res(i), res(i+1)) ;
                
            otherwise
                error('Unknown layer type ''%s''.', l.type) ;
        end
        
        % optionally forget intermediate results
        needsBProp = doder && i >= backPropLim;
        forget = opts.conserveMemory && ~needsBProp ;
        if i > 1
            lp = net.layers{i-1} ;
            % forget RELU input, even for BPROP
            forget = forget && (~needsBProp || (strcmp(l.type, 'snn_relu_th') && ~lp.precious)) ;
            forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
            forget = forget && ~lp.precious ;
        end
        if forget
            res(i).x = [] ;
        end
        
        if gpuMode && opts.sync
            wait(gpuDevice) ;
        end
        res(i).time = toc(res(i).time) ;
    end
    snnopt.islast = false;
    
    if gpuMode
        
        scores = squeeze((res(end-1).x)) ;
        [~, best] = max(scores) ;
        net.best(:,t) = single(gather(best));
        
    end
    
    %     res.best = best;
    %         if numel(best)>1
    %             fprintf('Time: %1.3fms \n', t/1000);
    %         else
    %             fprintf('Time: %1.3fms best: %2i   \n', t/1000, best);
    %         end
    %     end
    
end


% vl_simplenn_display(net);
% export sum of spikes
% if gpuMode
%     snet = struct(); %

%     for i=2:n
%
%         if strcmp(net.layers{i}.type,'snn_relu_th')||strcmp(net.layers{i}.type,'snn_relu_th_MAX')
%
%             snet.layers{i}.type = 'snn_relu_th';
%             snet.layers{i}.sum_spikes = gather(net.layers{i}.sum_spikes);
%
%         end
%     end
%     snet.layers{1}.type = 'snn_relu_th';
%     snet.layers{1}.sum_spikes = gather(net.layers{1}.sum_spikes);
% else
%     snet = net;
% end


if (t_opts.do_arausal)
    net.arausal = zeros(size(snnopt.arausal));
    net.arausal(not(snnopt.arausal)) = snnopt.theta0;
    net.arausal(snnopt.arausal) = snnopt.theta0_a;        
end


end


%% -------------------------------------------------------------------------
%                                                           Reset
% -------------------------------------------------------------------------
function l = reset_snn_layer(l)

sx = size(l.y);

% ltheta0 = gpuArray(snnopt.theta0);
% lthMag = gpuArray(snnopt.thMag);

correctly_sized_zeros = zeros(sx,'single','gpuArray');

l.y = correctly_sized_zeros;
l.yhat = correctly_sized_zeros;
l.th1 = correctly_sized_zeros;
if l.sTh>0
    l.th2 = correctly_sized_zeros;
    if l.tTh>0
        l.th3 = correctly_sized_zeros;
    end
end

% reset theta
% l.theta =  l.scaleth;

l.theta =  bsxfun(@times, ones(sx,'single','gpuArray'), l.theta0);
% l.scaleth = repmat(ltheta0,1,1,1,sx(4));
% l.scalethMag = repmat(lthMag,1,1,1,sx(4));

l.spikes = correctly_sized_zeros;
l.epsi = correctly_sized_zeros;
l.epsi_b = correctly_sized_zeros;
% l.prevspike = correctly_sized_zeros;

if isfield(l,{'o_epsif'})
    l.o_epsif = correctly_sized_zeros;
end

% free un-used
% ltheta0 = [];
% lthMag = [];


end

% -------------------------------------------------------------------------

function l = reset_snn_MAX_layer(l)

sx = size(l.y);
sx_ = size(l.yhat);


% ltheta0 = gpuArray(snnopt.theta0);
% lthMag = gpuArray(snnopt.thMag);

correctly_sized_zeros = zeros(sx,'single','gpuArray');
max_sized_zeros = zeros(sx_,'single','gpuArray');

l.y = correctly_sized_zeros;
l.yhat = max_sized_zeros;
l.th1 = max_sized_zeros;
if l.sTh>0
    l.th2 = max_sized_zeros;
    if l.tTh>0
        l.th3 = max_sized_zeros;
    end
end

% l.theta =  l.scaleth;

l.theta =  bsxfun(@times, ones(size(max_sized_zeros),'single','gpuArray'), l.theta0);
% l.scaleth = repmat(ltheta0,1,1,1,sx(4));
% l.scalethMag = repmat(lthMag,1,1,1,sx(4));

l.spikes = max_sized_zeros;
l.epsi = correctly_sized_zeros;
l.epsi_b = correctly_sized_zeros;
% l.prevspike = max_sized_zeros;

if isfield(l,{'o_epsif'})
    l.o_epsif = max_sized_zeros;
end

% free un-used
% ltheta0 = [];
% lthMag = [];


end

% -------------------------------------------------------------------------

function spikesize = compute_scalefactor(theta_0)


% theta_0=0.5;
h         = single(1);
m_f       = single(theta_0);
tau_gamma = single(15);
tau_eta   = single(50);
theta0    = single(theta_0);


c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;

c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1) - h/2;


y1 = max(0,h./(exp((c1 + c2)./(c3 + c4)) - 1) - c0 );
spikesize = single(1/y1);

end



% 
% % -------------------------------------------------------------------------
% function net = rest_snn_a(net, a)
% 
% 
% 
% for i=1:numel(net.layers)
%     
%     if strcmp(net.layers{i}.type,'snn_sumtanh')||...
%             strcmp(net.layers{i}.type,'snn_relu_th')||...
%             strcmp(net.layers{i}.type,'snn_sumtanh_bnorm')                
%         if isfield(net.layers{i},{'y'})
%             
%             net.layers{i} = reset_snn_layer_a(net.layers{i},a);
%             
%         end
%     end
%     if strcmp(net.layers{i}.type,'snn_sumtanh_MAX')||...
%             strcmp(net.layers{i}.type,'snn_relu_th_MAX')||...
%             strcmp(net.layers{i}.type,'snn_sumtanh_bnorm_max')
%                 
%         if isfield(net.layers{i},{'y'})
%             
%             net.layers{i} = reset_snn_layer_a(net.layers{i},a);
%             
%         end
%     end
%     
%     
% end
% 
% 
% 
% 
% end
% 
% % -------------------------------------------------------------------------
% function l = reset_snn_layer_a(l,a)
% 
% l.epsi(:,:,:,a) = l.epsi(:,:,:,a)*0; 
% l.y(:,:,:,a) = l.y(:,:,:,a)*0;
% l.yhat(:,:,:,a) = l.yhat(:,:,:,a)*0;
% l.th1(:,:,:,a) = l.th1(:,:,:,a)*0;
% 
% 
% if isfield(l,{'o_epsif'})
%     l.o_spikes(:,:,:,a) = l.o_spikes(:,:,:,a)*0;
%     l.o_epsif(:,:,:,a) = l.o_epsif(:,:,:,a)*0;
% end
% 
% 
% 
% end



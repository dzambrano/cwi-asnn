function [res, net] = snn_simplenn_opt(net, x, dzdy, res, t_opts, varargin)
%SNN_SIMPLENN  Evaluate a Simple SNN spiking-neural-network.
%   This code relies on vl_simplenn.m
%   Please refer to that file for help on the main functions.
%   This code has been modified to evaluate spiking neural networks.
%   Every relu layer and the input image will be converted on snn_relu.

% opts = t_opts;
% x = im_1;

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts = vl_argparse(opts, varargin);

if ~strcmp(net.layers{1}.type,'relu_th')
    % add an input spiking layer
    firstlayer = cell(1,1); %struct('type', 'snn_relu'));
    firstlayer{1} = struct('type', 'relu_th', 'name', 'relu1', 'precious', 1==2, 'leak', 0) ;
    net.layers = [firstlayer, net.layers] ;
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
else
  cudnn = {'NoCuDNN'} ;
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
    'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
  res(1).x = x ;
end

%% -------------------------------------------------------------------------
%                                                              Init SNN
% -------------------------------------------------------------------------
% x = im_1; size(x)

snnopt.gpuMode = gpuMode;

snnopt.theta0 = single(t_opts.threshold); %theta0 = 0.1;
snnopt.etaTau = single(50);
snnopt.thMag1 = single(0.1);
snnopt.thTau1 = single(15);
snnopt.thMag2 = single(0*0.01);
snnopt.thTau2 = single(100);
snnopt.thMag3 = single(0*0.005);
snnopt.thTau3 = single(500);
snnopt.thMag = single(t_opts.multip_nob_o);
snnopt.facMag = snnopt.thMag - t_opts.multip_nob_o;

thCorr = single(1/(snnopt.thMag1+snnopt.thMag2+snnopt.thMag3));
snnopt.thMag1c = snnopt.thMag1*thCorr;
snnopt.thMag2c = snnopt.thMag2*thCorr;
snnopt.thMag3c = snnopt.thMag3*thCorr;

snnopt.dth1 = exp(-1/snnopt.thTau1);
snnopt.dth2 = exp(-1/snnopt.thTau2);
snnopt.dth3 = exp(-1/snnopt.thTau3);
snnopt.dEta = exp(-1/snnopt.etaTau);

snnopt.mISI = single(1.25);
snnopt.ISI = single(55);
tauSig = single(.0025);
snnopt.a1 = exp (-0.001/tauSig);

tauSig2 = single(.025);
snnopt.a2 = exp (-0.001/tauSig2);
snnopt.islast = false;
t_duration = 1000*t_opts.duration;


% t_opts.thresLayer   = res_t.mstdx;
snnopt.thresLayer = t_opts.thresLayer;
snnopt.ismax = 0;
snnopt.cudnn = cudnn;


lambdaFac = 5;
thetaFac = 5.0;

% snet.layers = {} ;

% find last, reset
for i=1:n
    if strcmp(net.layers{i}.type,'relu')
        
        nrelulast = i;
%         snet.layers{i} = net.layers{i};
%     else
%         
%         snet.layers{i}.type = net.layers{i}.type;
        
    end

    if gpuMode %save initialization time for gpus
        if strcmp(net.layers{i}.type,'snn_relu_th')
            
            nrelulast = i;
            if isfield(net.layers{i},{'y'})
                
%                 lth = snnopt.thresLayer(i).fitWDistA;
%                 snnopt.theta0 = lth/lambdaFac;
%                 snnopt.thMag = thetaFac*snnopt.theta0;
                net.layers{i} = reset_snn_layer(net.layers{i});
                
            end
        end
        if strcmp(net.layers{i}.type,'snn_relu_th_MAX')
            
            nrelulast = i;
            if isfield(net.layers{i},{'y'})
                
%                 lth = snnopt.thresLayer(i+1).fitWDistA;
%                 snnopt.theta0 = lth/lambdaFac;
%                 snnopt.thMag = thetaFac*snnopt.theta0;
                net.layers{i} = reset_snn_MAX_layer(net.layers{i});
                
            end
        end
    end
end

% vl_simplenn_display(snet);

%% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for t=1:t_duration
    % Create spikes from the input images t=t+1
    %   (for all images in parallel)
    
    snnopt.isfirst = 1;
    l = net.layers{1} ;
    lth = snnopt.thresLayer(1).fitWDistA.*snnopt.thresLayer(1).fitWDistB;
    res(1).time = tic ;
    mask_x = res(1).x<0; % (only for the first layer) we consider negative spikes

    bias = single(zeros(size(res(1).x)));
    
    if ~isfield(l,{'y'})
        
        snnopt.theta0 = lth/lambdaFac;
        snnopt.thMag = thetaFac*snnopt.theta0;
    end
    
    
    if gpuMode     
        [res(2).x, l] = snn_relu_th_gpu(abs(res(1).x), l, bias, t, snnopt) ;
    else
        [res(2).x, l] = snn_relu_th(abs(res(1).x), l, bias, t, snnopt) ;       
    end
    net.layers{1} = l;
    
    res(2).x(mask_x) = -res(2).x(mask_x);
    res(1).time = toc(res(1).time) ;
    
    snnopt.isfirst = 0;
    
    for i=2:n
        
        
        if opts.skipForward, break; end;
        l = net.layers{i} ;
        
        res(i).time = tic ;
        
        
        if strcmp(l.type,'conv')
            bias = l.weights{2};
            if snnopt.gpuMode
                                
                l.weights{2} = 0*l.weights{2};
                
            else
                
                l.weights{2} = single(zeros(size(l.weights{2})));
                bias = gather(bias);
                
            end
            
        end
        
        if strcmp(l.type,'relu')
        
            l.type = 'snn_relu_th';
            
            if strcmp(net.layers{i+1}.type, 'pool') && strcmp(net.layers{i+1}.method, 'max')
                
                l.type = 'snn_relu_th_MAX';
                
            end
           
        end

        
        if i == nrelulast
            % the last-spikingNeuron output is analog
            snnopt.islast = true;  
        end
        
      
        switch l.type
            case 'conv'
                res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                    'pad', l.pad, ...
                    'stride', l.stride, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
                
            case 'convt'
                res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                    'crop', l.crop, ...
                    'upsample', l.upsample, ...
                    'numGroups', l.numGroups, ...
                    l.opts{:}, ...
                    cudnn{:}) ;
                
            case 'pool'
                if ~snnopt.ismax
                    res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
                        'pad', l.pad, 'stride', l.stride, ...
                        'method', l.method, ...
                        l.opts{:}, ...
                        cudnn{:}) ;
                else
                    res(i+1).x = res(i).x;
                    snnopt.ismax = 0;
                end
                
            case {'normalize', 'lrn'}
                res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
                
            case 'softmax'
                res(i+1).x = vl_nnsoftmax(res(i).x) ;
%                 vl_nnsoftmax(snet.layers{end}.o_epsif)
                
            case 'loss'
                res(i+1).x = vl_nnloss(res(i).x, l.class) ;
                
            case 'softmaxloss'
                
                res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
                
            case 'relu'
                if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
                res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
            
            case 'snn_relu'
                [res(i+1).x, l] = snn_relu(res(i).x, l, t, snnopt) ;
                net.layers{i} = l;  
                
            case 'snn_relu_th'
                if ~isfield(l,{'y'})
                    lth = snnopt.thresLayer(i).fitWDistA.*snnopt.thresLayer(i).fitWDistB;
                    snnopt.theta0 = lth/lambdaFac;
                    snnopt.thMag = thetaFac*snnopt.theta0;
                end
                if gpuMode 
%                      lg = net.layers{i} ;
%         
%             lg.type = 'snn_relu_th';
%                     im_1g = gpuArray(res(i).x) ;
%                     biasg = gpuArray(bias) ;
%                     [bbb, lg] = snn_relu_th_gpu(im_1g, lg, biasg, t, snnopt) ;
                    
                    [res(i+1).x, l] = snn_relu_th_gpu(res(i).x, l, bias, t, snnopt) ;
                else
%                     l = net.layers{i} ;
%                     
%                     l.type = 'snn_relu_th';

                    [res(i+1).x, l] = snn_relu_th(res(i).x, l, bias, t, snnopt) ;
                end
                net.layers{i} = l;          
                
            case 'snn_relu_th_MAX'
                
                l1 = net.layers{i+1} ;
                if ~isfield(l,{'y'})
                    lth = snnopt.thresLayer(i+1).fitWDistA.*snnopt.thresLayer(i+1).fitWDistB;
                    snnopt.theta0 = lth/lambdaFac;
                    snnopt.thMag = thetaFac*snnopt.theta0;
                end
                if gpuMode
%                     [bbb, lg] = snn_relu_th_max_gpu(im_1g, lg, l1, biasg, t, snnopt) ;

                    [res(i+1).x, l] = snn_relu_th_max_gpu(res(i).x, l, l1, bias, t, snnopt) ;
                else
                    [res(i+1).x, l] = snn_relu_th_max(res(i).x, l, l1, bias, t, snnopt) ;
                end
                net.layers{i} = l;
                snnopt.ismax = 1;
                
            case 'sigmoid'
                res(i+1).x = vl_nnsigmoid(res(i).x) ;
                
            case 'noffset'
                res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
                
            case 'spnorm'
                res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
                
            case 'dropout'
                if testMode
                    res(i+1).x = res(i).x ;
                else
                    [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
                end
                
            case 'bnorm'
                if testMode
                    res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
                else
                    res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
                end
                
            case 'pdist'
                res(i+1).x = vl_nnpdist(res(i).x, l.class, l.p, ...
                    'noRoot', l.noRoot, ...
                    'epsilon', l.epsilon, ...
                    'aggregate', l.aggregate, ...
                    'instanceWeights', l.instanceWeights) ;
                
           
                
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
    if ~gpuMode
        scores = squeeze(gather(res(end-1).x)) ;
        [~, best] = max(scores) ;
        if numel(best)>1
            fprintf('Time: %1.3fms \n', t/1000);
        else
            fprintf('Time: %1.3fms best: %2i   \n', t/1000, best);
        end
    end
    
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
l.theta =  l.scaleth;
% l.theta =  bsxfun(@times, ones(sx,'single','gpuArray'), ltheta0);
% l.scaleth = repmat(ltheta0,1,1,1,sx(4));
% l.scalethMag = repmat(lthMag,1,1,1,sx(4));

l.spikes = correctly_sized_zeros;
l.epsi = correctly_sized_zeros;
l.epsi_b = correctly_sized_zeros;
l.prevspike = correctly_sized_zeros;

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

l.theta =  l.scaleth;

% l.theta =  bsxfun(@times, ones(size(max_sized_zeros),'single','gpuArray'), ltheta0);  
% l.scaleth = repmat(ltheta0,1,1,1,sx(4));
% l.scalethMag = repmat(lthMag,1,1,1,sx(4));

l.spikes = max_sized_zeros;
l.epsi = correctly_sized_zeros;
l.epsi_b = correctly_sized_zeros;
l.prevspike = max_sized_zeros;

if isfield(l,{'o_epsif'})
    l.o_epsif = max_sized_zeros;
end

% free un-used
% ltheta0 = [];
% lthMag = [];


end


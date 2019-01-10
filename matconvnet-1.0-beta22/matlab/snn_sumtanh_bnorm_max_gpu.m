function [res_x, l] = snn_sumtanh_bnorm_max_gpu(x, l, l1, l2, bias, snnopt)
% %
% %#codegen
% code.inline('never')
%


if ~isfield(l,{'y'})
    
    l.theta0 = gpuArray(snnopt.theta0); 
    
    l.dEta = gpuArray(snnopt.dEta);
    
    l.dth1 = gpuArray(snnopt.dth1);
    
    l.sTh = 0; % second component
    l.tTh = 0; % third component
                    
    correctly_sized_zeros = zeros(size(x),'single','gpuArray');

    l.y = correctly_sized_zeros;
    
    max_sized_zeros = vl_nnpool(l.y, l2.pool, ...
        'pad', l2.pad, 'stride', l2.stride, ...
        'method', 'max', ...
        l2.opts{:}, ...
        snnopt.cudnn{:}) ;
    
    l.yhat = max_sized_zeros;
    l.th1 = max_sized_zeros;
    
    if(~snnopt.isfirst)
        
        l.bias = correctly_sized_zeros;
        l.bias = fun_init_bias(l.bias, bias, size(x));
        
    end
    
    %         l.normk = gpuArray(single(4.6845));%4.6845%12.3577
    l.normk = gpuArray(snnopt.spikesize);
    
    l.theta =  bsxfun(@times, ones(size(max_sized_zeros),'single','gpuArray'), l.theta0);
        
    l.spikes = max_sized_zeros;
    l.epsi = correctly_sized_zeros;
%     l.epsi_b = correctly_sized_zeros;
    l.sum_spikes = max_sized_zeros;


    if(snnopt.islast)
        l.o_epsif = max_sized_zeros;
        l.o_spikes = correctly_sized_zeros;

    end
    
    
end


% Add input

if(snnopt.isfirst)
    l.y = x;
else
    
    
    l.epsi = arrayfun(@update_in, l.epsi, l.dEta, x, l.normk);
   
    l.y = arrayfun(@fun_exp_filter, l.y, snnopt.a1, l.epsi + l.bias);
    
end


bnorm_y = vl_nnbnorm(l.y, l1.weights{1}, l1.weights{2}, ...
    'moments', l1.weights{3}, ...
    'epsilon', l1.epsilon, ...
    snnopt.bnormCudnn{:}) ;

% max pooling
max_y = vl_nnpool(bnorm_y, l2.pool, ...
        'pad', l2.pad, 'stride', l2.stride, ...
        'method', 'max', ...
        l2.opts{:}, ...
        snnopt.cudnn{:}) ;



% Check for spiking
v = arrayfun(@check_spikes, l.theta, l.yhat, max_y);

% Add spikes
[l.spikes, l.sum_spikes] = arrayfun(@add_spikes, v, l.sum_spikes, l.theta);

% yhat evolves
l.yhat = arrayfun(@update_yhat, l.yhat, l.spikes, l.dEta);

% update theta components:
[l.th1, l.theta] =  arrayfun(@update_th, l.th1, l.dth1, l.spikes, l.theta0);


if(snnopt.islast)
    
    
    l.o_spikes = l.o_spikes + single(v).*l.normk;    
    l.o_spikes =  l.o_spikes * l.dEta;    
    l.o_epsif = arrayfun(@fun_exp_filter,l.o_epsif, snnopt.a2, l.o_spikes);    
    res_x = gpuArray(l.o_epsif) ;
    l.o_v = single(v);
    
else
    
    res_x = single(v);
    
end


end


% -------------------------------------------------------------------------
function lbias = fun_init_bias(lbias, bias, sx)

for i=1:numel(bias)
    
    mb = ones(sx(1), sx(2),'single','gpuArray').*bias(i); % MC
    if size(sx,2)>3 lbias(:,:,i,:)=repmat(mb,1,1,1,sx(4));
    else lbias(:,:,i,:)=repmat(mb,1,1,1,1);
    end
    
end

end

% -------------------------------------------------------------------------
function y = fun_exp_filter(y, a, in)

y = y + (1-a)*(in-y); %MC

end



% -------------------------------------------------------------------------
function epsi = update_in(epsi, dEta, x, normk)

epsi = epsi * dEta + x * normk;

end


% -------------------------------------------------------------------------
function [th1, theta] = update_th(th1, dth1, spikes, theta0)

th1 = (th1 + spikes * theta0) * dth1;
theta = th1 + theta0;

end


% -------------------------------------------------------------------------
function yhat = update_yhat(yhat, spikes, dEta)


yhat =  (yhat + spikes) * dEta;

end


% -------------------------------------------------------------------------
function v = check_spikes(theta, yhat, y)

di = theta*0.5 + yhat;
v = y>di;


end


% -------------------------------------------------------------------------
function [spikes, sum_spikes] = add_spikes(v, sum_spikes, theta)

spikes =  v * theta;
sum_spikes = sum_spikes + single(v);

end

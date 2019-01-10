function [res_x, l] = snn_relu_th_max_gpu(x, l, l1, bias, t, snnopt)
% % 
% %#codegen
% code.inline('never')
% 


if ~isfield(l,{'y'})
   
    ltheta0 = gpuArray(snnopt.theta0);
    lthMag = gpuArray(snnopt.thMag);
    
    l.ISI = gpuArray(snnopt.ISI);
    l.mISI = gpuArray(snnopt.mISI);
    l.dEta = gpuArray(snnopt.dEta);
    
    l.fmISI = (l.mISI-1)/l.ISI;
    
    l.dth1 = gpuArray(snnopt.dth1);
    l.dth2 = gpuArray(snnopt.dth2);
    l.dth3 = gpuArray(snnopt.dth3);
    
    l.thMag1c = gpuArray(snnopt.thMag1c);
    l.thMag2c = gpuArray(snnopt.thMag2c);
    l.thMag3c = gpuArray(snnopt.thMag3c);
    
    l.sTh = 0; % second component
    l.tTh = 0; % third component
    if snnopt.thMag2c > 0       
        l.sTh = 1;        % second component
        if snnopt.thMag3c > 0            
            l.tTh = 1;    % third component
        end
        
    end
                    
    correctly_sized_zeros = zeros(size(x),'single','gpuArray');
    l.y = correctly_sized_zeros;
    
    max_sized_zeros = vl_nnpool(l.y, l1.pool, ...
        'pad', l1.pad, 'stride', l1.stride, ...
        'method', 'max', ...
        l1.opts{:}, ...
        snnopt.cudnn{:}) ;
    
    l.yhat = max_sized_zeros;
    l.th1 = max_sized_zeros;
    if l.sTh>0
        l.th2 = max_sized_zeros;
        if l.tTh>0
            l.th3 = max_sized_zeros;
        else
            l.th3 = [];
        end
    else
        l.th2 = [];
        l.th3 = [];
    end
    
    if(~snnopt.isfirst)
        
        l.bias = correctly_sized_zeros;        
        l.bias = fun_init_bias(l.bias, bias, size(x));
    
    end
    
%     l.theta =  bsxfun(@times, ones(size(max_sized_zeros),'single','gpuArray'), ltheta0);
    %     l.thMag =  bsxfun(@times, ones(size(x),'single','gpuArray'), lthMag);
    %     l.thMagO =  bsxfun(@times, ones(size(x),'single','gpuArray'), lthMag);
    l.theta = repmat(ltheta0,1,1,1,size(max_sized_zeros,4));
    l.scaleth = l.theta;
    l.scalethMag = repmat(lthMag,1,1,1,size(max_sized_zeros,4));
    
    l.spikes = max_sized_zeros;
    l.epsi = correctly_sized_zeros;
    l.epsi_b = correctly_sized_zeros;
    l.sum_spikes = max_sized_zeros;
    l.prevspike = max_sized_zeros;

    if(snnopt.islast)
        l.o_epsif = max_sized_zeros;
    end
    
    % free un-used
    ltheta0 = [];
    lthMag = [];

end


% Add input 

if(snnopt.isfirst)
    l.y = x;
else
    
    l.epsi = l.epsi*l.dEta + x; %MC
    l.epsi_b = l.epsi + l.bias;    
    l.y = arrayfun(@fun_exp_filter, l.y, snnopt.a1, l.epsi_b);
end

% max pooling
max_y = vl_nnpool(l.y, l1.pool, ...
        'pad', l1.pad, 'stride', l1.stride, ...
        'method', 'max', ...
        l1.opts{:}, ...
        snnopt.cudnn{:}) ;
    
% Check for spiking
v = fun_check_spike(max_y, l.yhat, l.theta);
% v = bsxfun(@gt,(l.y - l.yhat),(l.theta/2));

l.sum_spikes = v+l.sum_spikes; 

% if a spike: add to approximation
deltaSpike = t - l.prevspike;

visi = deltaSpike > l.ISI;
nvisi = 1-visi;

isifact = arrayfun(@isif_out,deltaSpike, visi, nvisi, l.mISI, l.fmISI);

visi = [];
nvisi = [];


l.spikes =  arrayfun(@fun_multy, v, l.theta, isifact);
%yhat evolves
l.yhat = l.yhat + l.spikes;   %CRITICAL

fmult =  bsxfun(@rdivide, v.*l.theta, l.scaleth);       %CRITICAL

% update theta components:
l.th1 = arrayfun(@fun_l_th_n,l.th1,fmult,l.scalethMag,l.thMag1c);
l.th1 =  bsxfun(@times, l.th1, l.dth1);
if l.sTh
    l.th2 = arrayfun(@fun_l_th_n,l.th2,fmult,l.scalethMag,l.thMag2c);
    l.th2 =  bsxfun(@times, l.th2, l.dth2);
    if l.tTh
        l.th3 = arrayfun(@fun_l_th_n,l.th3,fmult,l.scalethMag,l.thMag3c);
        l.th3 =  bsxfun(@times, l.th3, l.dth3);
    end
end

nv = 1-v;
l.prevspike = arrayfun(@fun_t, v, nv, t, l.prevspike);

% evolve yhat
l.yhat =  bsxfun(@times, l.yhat, l.dEta);
% l.theta = arrayfun(@sum_theta,l.th1,l.th2,l.th3,l.scaleth);
l.theta = sum_theta(l.th1,l.th2,l.th3,l.scaleth);

if(snnopt.islast)
    
    l.o_epsif = arrayfun(@fun_exp_filter,l.o_epsif, snnopt.a2, l.yhat);
    
    if snnopt.gpuMode
        res_x = gpuArray(l.o_epsif) ;
    else
        res_x = single(l.o_epsif);        
    end
else
    
    if snnopt.gpuMode
        res_x = l.spikes ;
    else
        res_x = single(l.spikes);
    end
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
function v = fun_check_spike(ly, lyhat, ltheta)

% di = bsxfun(@minus,ly,lyhat);
di = ly - lyhat;
lt2 = ltheta*0.5;
% lt2 = bsxfun(@times,ltheta,0.5);
v = di>lt2;

end

% -------------------------------------------------------------------------
function isifact = isif_out(deltaSpike, visi, nvisi, mISI, fmISI)

isifact = (visi.*mISI) +  nvisi.*(1 + deltaSpike.*fmISI); %CRITICAL

end

% -------------------------------------------------------------------------
function theta = sum_theta(th1,th2,th3,scaleth)

if isempty(th2)
    theta = th1 + scaleth;    
elseif isempty(th3)    
    theta = th1 + th2 + scaleth;
else
    theta = th1 + th2 + th3 + scaleth;    
end

end

% -------------------------------------------------------------------------
function l_theta = fun_l_th_n(l_th_n, fmult, scalethMag, thMag_n_c)

l_theta = l_th_n + fmult.*scalethMag*thMag_n_c;

end

% -------------------------------------------------------------------------
function prevspike = fun_t(v,nv, t, pspike)

prevspike = v.*t + nv.*pspike;

end

% -------------------------------------------------------------------------
function y = fun_exp_filter(y, a, in)

y = y + (1-a)*(in-y); %MC

end


% -------------------------------------------------------------------------
function lspikes = fun_multy(v, ltheta, isif)

lspikes = v*ltheta*isif; %CRITICAL

end
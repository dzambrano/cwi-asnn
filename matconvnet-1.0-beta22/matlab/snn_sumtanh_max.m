function [res_x, l] = snn_sumtanh_max(x, l, l1, bias, t, snnopt)
% % 
%
% 


if ~isfield(l,{'y'})
    %
    correctly_sized_zeros = single(zeros(size(x)));
    l.y = correctly_sized_zeros;
    
    max_sized_zeros = vl_nnpool(l.y, l1.pool, ...
        'pad', l1.pad, 'stride', l1.stride, ...
        'method', 'max', ...
        l1.opts{:}, ...
        snnopt.cudnn{:}) ;
    
%     %
    % generate spikes only after max pooling
    l.yhat = max_sized_zeros;
    l.th1 = max_sized_zeros;
    l.th2 = max_sized_zeros;
    l.th3 = max_sized_zeros;

    if(~snnopt.isfirst)
        l.bias = correctly_sized_zeros;
        for i=1:numel(bias)
            
            mb = single(ones(size(x,1),size(x,2))).*bias(i);
            l.bias(:,:,i,:)=repmat(mb,1,1,1,size(x,4));
            
        end
    end
    
%     l.theta =  snnopt.theta0.*ones(size(x));
    l.theta =  bsxfun(@times, ones(size(max_sized_zeros)), snnopt.theta0);      
    l.thMag =  bsxfun(@times, ones(size(max_sized_zeros)), snnopt.thMag);
    l.thMagO =  bsxfun(@times, ones(size(max_sized_zeros)), snnopt.thMag);
    l.scaleth = repmat(snnopt.theta0,1,1,1,size(max_sized_zeros,4));
    l.scalethMag = repmat(snnopt.thMag,1,1,1,size(max_sized_zeros,4));

    l.normk = (single(4.0943));

    l.spikes = max_sized_zeros;
    l.epsi = correctly_sized_zeros;
    l.epsi_b = correctly_sized_zeros;
    l.sum_spikes = max_sized_zeros;
    l.prevspike = max_sized_zeros;
%   
    if(snnopt.islast)
        l.o_epsif = max_sized_zeros;
    end


end


if(snnopt.isfirst)
    l.y = x;
else
    % add bias
    l.epsi = l.epsi*snnopt.dEta + x;
    l.epsi_b = l.epsi + l.bias;    
    l.y = l.y + (1-snnopt.a1)*(l.epsi_b-l.y);
    
end

% max pooling
max_y = vl_nnpool(l.y, l1.pool, ...
        'pad', l1.pad, 'stride', l1.stride, ...
        'method', 'max', ...
        l1.opts{:}, ...
        snnopt.cudnn{:}) ;
    
% Check for spiking
v = (max_y - l.yhat)>(l.theta/2);
l.sum_spikes(v) = l.sum_spikes(v)+1;
% if a spike: add to approximation
deltaSpike = t - l.prevspike;

correctly_sized_zeros = single(zeros(size(max_y)));
isifact = correctly_sized_zeros;

visi = deltaSpike > snnopt.ISI;
isifact(visi) = snnopt.mISI;
isifact(not(visi)) = 1 + deltaSpike(not(visi))*(snnopt.mISI-1)/snnopt.ISI;

%yhat evolves
l.yhat(v) = l.yhat(v) + ...
    l.theta(v).*isifact(v);

l.spikes = correctly_sized_zeros;
% l.spikes(v) = 1;
l.spikes(v) = l.y(v);


fmult = correctly_sized_zeros;
fmult(v) =  bsxfun(@rdivide, l.theta(v), l.scaleth(v));      

% fmult(v) = l.theta(v)./snnopt.theta0;

% update theta components:
l.th1 = l.th1 + fmult.*l.scalethMag*snnopt.thMag1c;
l.th2 = l.th2 + fmult.*l.scalethMag*snnopt.thMag2c;
l.th3 = l.th3 + fmult.*l.scalethMag*snnopt.thMag3c;

l.prevspike(v) = t;

% evolve yhat
l.yhat = l.yhat*snnopt.dEta;
l.th1 = l.th1*snnopt.dth1;
l.th2 = l.th2*snnopt.dth2;
l.th3 = l.th3*snnopt.dth3;
l.theta = ...
    l.th1+l.th2+l.th3 + l.scaleth;


%  l.epsi = l.epsi+(1-snnopt.a1)*(l.yhat-l.epsi); %THIS

if(snnopt.islast)
    
    l.o_epsif = l.o_epsif+(1-snnopt.a1)*(l.yhat-l.o_epsif);
    res_x = single(l.o_epsif);
    if snnopt.gpuMode
        res_x = gpuArray(res_x) ;
    end
else
    res_x = single(v)/l.normk;
    if snnopt.gpuMode
        res_x = gpuArray(res_x) ;
    end
%      res_x = single(l.epsi); %THIS
end


end
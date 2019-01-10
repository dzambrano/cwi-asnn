function [res_x, l] = snn_sumtanh(x, l, bias, t, snnopt)
% % 
% %#codegen
% code.inline('never')
% 


if ~isfield(l,{'y'})
    %
%    
%     snnopt.theta0 = lth/2000;
%     snnopt.thMag = 1*snnopt.theta0;
    %
    
    correctly_sized_zeros = single(zeros(size(x)));
    l.y = correctly_sized_zeros;
    l.yhat = correctly_sized_zeros;
    l.th1 = correctly_sized_zeros;
    l.th2 = correctly_sized_zeros;
    l.th3 = correctly_sized_zeros;
    

    if(~snnopt.isfirst)
        l.bias = correctly_sized_zeros;
        for i=1:numel(bias)
            
            mb = single(ones(size(x,1),size(x,2))).*bias(i);
            l.bias(:,:,i,:)=repmat(mb,1,1,1,size(x,4));
            
        end
    end
    
%     l.theta =  snnopt.theta0.*ones(size(x));
    l.theta =  bsxfun(@times, ones(size(x)), snnopt.theta0);      
    l.thMag =  bsxfun(@times, ones(size(x)), snnopt.thMag);
    l.thMagO =  bsxfun(@times, ones(size(x)), snnopt.thMag);
    l.scaleth = repmat(snnopt.theta0,1,1,1,size(x,4));
    l.scalethMag = repmat(snnopt.thMag,1,1,1,size(x,4));

    l.normk = (single(4.0943));

    l.spikes = correctly_sized_zeros;
    l.epsi = correctly_sized_zeros;
    l.epsi_b = correctly_sized_zeros;
    l.sum_spikes = correctly_sized_zeros;
    l.prevspike = correctly_sized_zeros;
%     
%     l.thMag =  snnopt.thMag.*ones(size(x));
%     l.thMagO =  snnopt.thMag.*ones(size(x));
    
    if(snnopt.islast)
        l.o_epsif = correctly_sized_zeros;
    end


end


% Add input %t=1


if(snnopt.isfirst)
    l.y = x;
else
    
    l.epsi = l.epsi*snnopt.dEta + x;
    l.epsi_b = l.epsi + l.bias;    
    l.y = l.y + (1-snnopt.a1)*(l.epsi_b-l.y);
    
end
% Check for spiking
v = (l.y - l.yhat)>(l.theta/2);
l.sum_spikes(v) = l.sum_spikes(v)+1;
% if a spike: add to approximation
deltaSpike = t - l.prevspike;

correctly_sized_zeros = single(zeros(size(x)));
isifact = correctly_sized_zeros;

visi = deltaSpike > snnopt.ISI;
isifact(visi) = snnopt.mISI;
isifact(not(visi)) = 1 + deltaSpike(not(visi))*(snnopt.mISI-1)/snnopt.ISI;

%yhat evolves
l.yhat(v) = l.yhat(v) + ...
    l.theta(v).*isifact(v);

l.spikes = correctly_sized_zeros;
l.spikes(v) = l.theta(v).*isifact(v);
% l.spikes(v) = 1;


fmult = correctly_sized_zeros;
fmult(v) =  bsxfun(@rdivide, l.theta(v), l.scaleth(v));      
% l.fmult = fmult;
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
% if snnopt.ismax
%     
%     
%     resx = vl_nnpool(l.yhat, snnopt.pool, ...
%         'pad', snnopt.pad, 'stride', snnopt.stride, ...
%         'method', 'max', ...
%         snnopt.opts{:}, ...
%         snnopt.cudnn{:}) ;
%     
%     
% end

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
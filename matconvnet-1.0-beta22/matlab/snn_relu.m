function [res_x, l] = snn_relu(x, l, t, snnopt)

if ~isfield(l,{'y'})
    %
   
%         snnopt.theta0 = snnopt;
%         snnopt.thMag = 0.1*snnopt.theta0;
    %
    
    
    correctly_sized_zeros = single(zeros(size(x)));
    l.y = correctly_sized_zeros;
    l.yhat = correctly_sized_zeros;
    l.th1 = correctly_sized_zeros;
    l.th2 = correctly_sized_zeros;
    l.th3 = correctly_sized_zeros;
    l.theta =  snnopt.theta0.*ones(size(x));
    l.spikes = correctly_sized_zeros;
    l.epsi = correctly_sized_zeros;
    l.sum_spikes = correctly_sized_zeros;
    l.prevspike = correctly_sized_zeros;
    
    l.thMag =  snnopt.thMag.*ones(size(x));
    l.thMagO =  snnopt.thMag.*ones(size(x));
    
    if(snnopt.islast)
        l.o_epsif = correctly_sized_zeros;
    end


end


% Add input %t=1


if(snnopt.isfirst)
    l.y = x;
else
    
    l.epsi = l.epsi*snnopt.dEta + x;    
    l.y = l.y + (1-snnopt.a1)*(l.epsi-l.y);
    
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

fmult = correctly_sized_zeros;
fmult(v) = l.theta(v)./snnopt.theta0;

% update theta components:
l.th1 = l.th1 + fmult*snnopt.thMag*snnopt.thMag1c;
l.th2 = l.th2 + fmult*snnopt.thMag*snnopt.thMag2c;
l.th3 = l.th3 + fmult*snnopt.thMag*snnopt.thMag3c;

l.prevspike(v) = t;

% evolve yhat
l.yhat = l.yhat*snnopt.dEta;
l.th1 = l.th1*snnopt.dth1;
l.th2 = l.th2*snnopt.dth2;
l.th3 = l.th3*snnopt.dth3;
l.theta = ...
    l.th1+l.th2+l.th3 + snnopt.theta0;

%  l.epsi = l.epsi+(1-snnopt.a1)*(l.yhat-l.epsi); %THIS

if(snnopt.islast)
    
    l.o_epsif = l.o_epsif+(1-snnopt.a1)*(l.yhat-l.o_epsif);
    res_x = single(l.o_epsif);
    
else
    res_x = single(l.spikes);
%      res_x = single(l.epsi); %THIS
end


end
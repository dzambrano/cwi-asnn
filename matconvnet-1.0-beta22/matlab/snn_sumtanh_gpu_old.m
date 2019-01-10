function [res_x, l] = snn_sumtanh_gpu(x, l, bias, t, snnopt)
% % 
% 

if ~isfield(l,{'y'})
   
    l.theta0 = gpuArray(snnopt.theta0); 
%     lthMag = gpuArray(snnopt.thMag);   
%     
%     ltheta0_a = gpuArray(snnopt.theta0_a); 
%     lthMag_a = gpuArray(snnopt.thMag_a);   
    
    %     l.ISI = gpuArray(snnopt.ISI);
    %     l.mISI = gpuArray(snnopt.mISI);
    %
    %     l.fmISI = (l.mISI-1)/l.ISI;
    
    l.dEta = gpuArray(snnopt.dEta);
    
    l.dth1 = gpuArray(snnopt.dth1);
%     l.dth2 = gpuArray(snnopt.dth2);
%     l.dth3 = gpuArray(snnopt.dth3);
%     
%     l.thMag1c = gpuArray(snnopt.thMag1c);
%     l.thMag2c = gpuArray(snnopt.thMag2c);
%     l.thMag3c = gpuArray(snnopt.thMag3c);
    
    l.sTh = 0; % second component
    l.tTh = 0; % third component
%     if snnopt.thMag2c > 0       
%         l.sTh = 1;        % second component
%         if snnopt.thMag3c > 0            
%             l.tTh = 1;    % third component
%         end
%         
%     end
                    
    correctly_sized_zeros = zeros(size(x),'single','gpuArray');

    l.y = correctly_sized_zeros;
    l.yhat = correctly_sized_zeros;
    l.th1 = correctly_sized_zeros;    
%     if l.sTh>0
%         l.th2 = correctly_sized_zeros;
%         if l.tTh>0
%             l.th3 = correctly_sized_zeros;
%         else
%             l.th3 = [];
%         end
%     else
%         l.th2 = [];
%         l.th3 = [];
%     end
%     
    if(~snnopt.isfirst)
        
        l.bias = correctly_sized_zeros;        
        l.bias = fun_init_bias(l.bias, bias, size(x));
    
    end
    
    l.normk = gpuArray(snnopt.spikesize); 
    l.normk_a = gpuArray(snnopt.spikesize_a); 

    l.theta =  bsxfun(@times, ones(size(x),'single','gpuArray'), l.theta0);
%         l.thMag =  bsxfun(@times, ones(size(x),'single','gpuArray'), lthMag);
    %     l.thMagO =  bsxfun(@times, ones(size(x),'single','gpuArray'), lthMag);
%     l.theta = repmat(ltheta0,1,1,1,size(x,4)); 
%     l.scaleth = l.theta;
    
%     l.scalethMag = l.theta.*l.theta; 

%     l.theta_a = repmat(ltheta0_a,1,1,1,size(x,4)); 
%     l.scaleth_a = l.theta_a;
    
    
%     l.scalethMag = repmat(lthMag,1,1,1,size(x,4));
%     l.scalethMag_a = repmat(lthMag_a,1,1,1,size(x,4));
    
    l.spikes = correctly_sized_zeros;
    l.epsi = correctly_sized_zeros;
    l.epsi_b = correctly_sized_zeros;
    l.sum_spikes = correctly_sized_zeros;
%     l.prevspike = correctly_sized_zeros;

    if(snnopt.islast)
        l.o_epsif = correctly_sized_zeros;
        l.o_spikes = correctly_sized_zeros;
    end
    
    % free un-used
%     ltheta0 = [];
%     lthMag = [];

end


% Add input 

if(snnopt.isfirst)
    l.y = x;
else
    
    l.epsi = l.epsi*l.dEta + x; %MC
    l.epsi_b = l.epsi + l.bias;    
    l.y = arrayfun(@fun_exp_filter, l.y, snnopt.a1, l.epsi_b);
end


% Check for spiking

di = bsxfun(@plus,l.theta*0.5,l.yhat);
v = l.y>di;

l.sum_spikes = l.sum_spikes + v; 

% spike size by current theta
l.spikes =  bsxfun(@times, v, l.theta);

% yhat evolves
l.yhat = l.yhat + l.spikes; %CRITICAL

% 
if snnopt.isarausal
    fmult =  bsxfun(@rdivide, l.spikes, l.scaleth_a);
    l.th1 = arrayfun(@fun_l_th_n,l.th1,fmult,l.scalethMag_a,l.thMag1c);%CRITICAL
   
else    
%     fmult =  bsxfun(@rdivide, l.spikes, l.scaleth);
%     l.th1 = arrayfun(@fun_l_th_n,l.th1,fmult,l.scalethMag,l.thMag1c);%CRITICAL
    
    l.th1 = l.th1 + bsxfun(@times, l.theta, l.theta0);
    
end

% update theta components:
l.th1 =  bsxfun(@times, l.th1, l.dth1);
% if l.sTh
%     l.th2 = arrayfun(@fun_l_th_n,l.th2,fmult,l.scalethMag,l.thMag2c);
%     l.th2 =  bsxfun(@times, l.th2, l.dth2);
%     if l.tTh
%         l.th3 = arrayfun(@fun_l_th_n,l.th3,fmult,l.scalethMag,l.thMag3c);
%         l.th3 =  bsxfun(@times, l.th3, l.dth3);
%     end
% end
% 



% evolve yhat
l.yhat =  bsxfun(@times, l.yhat, l.dEta);

if snnopt.isarausal
    l.theta = sum_theta(l.th1,l.th2,l.th3,l.scaleth_a);
else    
    %     l.theta = sum_theta(l.th1,l.th2,l.th3,l.scaleth);
    l.theta = l.theta0 + l.th1;
end

if(snnopt.islast)
    
    %     l.o_epsif = arrayfun(@fun_exp_filter,l.o_epsif, snnopt.a2, l.yhat);
    if snnopt.isarausal
        l.o_spikes = l.o_spikes + single(v).*l.normk_a;
    else        
        l.o_spikes = l.o_spikes + single(v).*l.normk;
    end
    l.o_spikes =  bsxfun(@times, l.o_spikes, l.dEta);
    
    l.o_epsif = arrayfun(@fun_exp_filter,l.o_epsif, snnopt.a2, l.o_spikes);

    if snnopt.gpuMode
        res_x = gpuArray(l.o_epsif) ;
    else
        res_x = single(l.o_epsif); 
    end
else    
    if snnopt.gpuMode  
        if snnopt.isarausal
            res_x = bsxfun(@times,single(v),l.normk_a);
        else
            res_x = bsxfun(@times,single(v),l.normk);
        end
        
    else
        res_x = single(v).*l.normk;
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

% % -------------------------------------------------------------------------
% function v = fun_check_spike(ly, lyhat, ltheta)
% 
% di = bsxfun(@minus,ly,lyhat);
% % di = ly - lyhat;
% lt2 = ltheta*0.5;
% % lt2 = bsxfun(@times,ltheta,0.5);
% v = di>lt2;
% 
% end

% % -------------------------------------------------------------------------
% function isifact = isif_out(deltaSpike, visi, nvisi, mISI, fmISI)
% 
% isifact = (visi.*mISI) +  nvisi.*(1 + deltaSpike.*fmISI); %CRITICAL
% 
% end

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

l_theta = l_th_n + fmult.*scalethMag.*thMag_n_c;

end

% % -------------------------------------------------------------------------
% function prevspike = fun_t(v,nv, t, pspike)
% 
% prevspike = v.*t + nv.*pspike;
% 
% end

% -------------------------------------------------------------------------
function y = fun_exp_filter(y, a, in)

y = y + (1-a)*(in-y); %MC

end


% % -------------------------------------------------------------------------
% function lspikes = fun_multy(v, ltheta, isif)
% 
% lspikes = v*ltheta*isif; %CRITICAL
% 
% end


% % -------------------------------------------------------------------------
% function v = fun_araSpikes(v, normkv)
% 
% for i = 1:numel(normkv)
%     v(:,:,:,i) = v(:,:,:,i)*normkv(i); 
% end
% 
% end



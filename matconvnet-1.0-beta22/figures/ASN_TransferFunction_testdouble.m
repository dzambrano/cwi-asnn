clc
clear all
close all
figure 
% hold on



%% Variance and MNOB
% 
% close all
t=1:500;

thMag1 = .1;
thTau1 = 15;
etaTau = 50;

theta0v = 0.1;

mnob = theta0v(1)*theta0v(1);
% mnob = [1*theta0v(1) 0.5*theta0v(2)];


inputC = 1;

firingratesnn = zeros(length(mnob),length(inputC));
variancesnn = zeros(length(mnob),length(inputC));
meansnn = zeros(length(mnob),length(inputC));


for imnob = 1:length(mnob)
    
    
    spiketime = [];
    frate = zeros(2,length(inputC));
    tfun = zeros(size(inputC));
    tfunmf = zeros(size(inputC,1),2);
    
    tauSig2 = 2.5;
    a1 = exp (-1/tauSig2);
    
    theta0 = theta0v(imnob);
    
    for imn = 1:length(inputC)
        
        % keep track of assigned theta:
        thetaTrack = ones(size(t))*theta0;
        
        % required for exp:
        theta = theta0;
        th0 = theta0;
        th1 = 0;
     
        
        thCorr = 1/(thMag1);
        thMag1c = thMag1*thCorr;
       
        dth1 = exp(-1/(thTau1));
        
        dEta = exp(-1/(etaTau));
        eta = 1;
        % signal:
        tstep = 1;
        input = zeros(1,length(t));
        input(tstep:end) = inputC(imn);
        
        yhat = zeros(size(input));
        spike = zeros(size(input));
        
        
        stimes1a = zeros(size(input));
        y = input;
        
        tauSig2 = .005;
        a2 = exp (-0.001/tauSig2);
        
        epsif = 0;
        
        epsifTrack = ones(size(t));
        
        mISI = 1;
        ISI = 55;
        prevspike = 0;
        
        % figure
        thMag = mnob(imnob);
        
        
        for time=1:length(t)
            
            
            v = y(time) - yhat(time);
            if v>(1*theta/2)
                stimes1a(time) = 1;
                % if a spike: add to approximation
                deltaSpike = time - prevspike;
                
                lenK = length(yhat(time:end));
                
                isifact  = mISI;

                dyhat = theta*isifact;
                yhat(time) = yhat(time) + dyhat;

                
                fmult = theta/theta0;

                
                spike(time) = spike(time) + 1;
                spiketime = [spiketime time];
                
                % update theta components:
                th1 = th1 + fmult*thMag;
                
                prevspike = time;
            end
            % evolve yhat
            yhat(time+1) = yhat(time)*dEta;
            spike(time+1) = spike(time)*dEta;
            th1 = th1*dth1;
           
            theta = th1 + theta0;
            thetaTrack(time) = theta;
            
            epsif = epsif+(1-a1)*(spike(time)-epsif);
            epsifTrack(time)=epsif;
            
            
        end
        
        
        
       
    end
    
    
firingratesnn(imnob,:)=tfun;
variancesnn(imnob,:)=tfunmf(:,2);
meansnn(imnob,:)=tfunmf(:,1);

end


subplot(2,1,1)
plot(t,yhat(1:end-1))
hold on

subplot(2,1,2)
plot(t,thetaTrack)
hold on


%% 
% close all
t=1:500;

thMag1 = .1;
thTau1 = 15;
etaTau = 50;

theta0v = 0.1/2;

mnob = theta0v(1)*theta0v(1);
% mnob = [1*theta0v(1) 0.5*theta0v(2)];


inputC = 1;

firingratesnn = zeros(length(mnob),length(inputC));
variancesnn = zeros(length(mnob),length(inputC));
meansnn = zeros(length(mnob),length(inputC));


for imnob = 1:length(mnob)
    
    
    spiketime = [];
    frate = zeros(2,length(inputC));
    tfun = zeros(size(inputC));
    tfunmf = zeros(size(inputC,1),2);
    
    tauSig2 = 2.5;
    a1 = exp (-1/tauSig2);
    
    theta0 = theta0v(imnob);
    
    for imn = 1:length(inputC)
        
        % keep track of assigned theta:
        thetaTrack = ones(size(t))*theta0;
        
        % required for exp:
        theta = theta0;
        th0 = theta0;
        th1 = 0;
     
        
        thCorr = 1/(thMag1);
        thMag1c = thMag1*thCorr;
       
        dth1 = exp(-1/(thTau1));
        
        dEta = exp(-1/(etaTau));
        eta = 1;
        % signal:
        tstep = 1;
        input = zeros(1,length(t));
        input(tstep:end) = inputC(imn);
        
        yhat = zeros(size(input));
        spike = zeros(size(input));
        
        
        stimes1a = zeros(size(input));
        y = input;
        
        tauSig2 = .005;
        a2 = exp (-0.001/tauSig2);
        
        epsif = 0;
        
        epsifTrack = ones(size(t));
        
        mISI = 1;
        ISI = 55;
        prevspike = 0;
        
        % figure
        thMag = mnob(imnob);
        
        
        for time=1:length(t)
            
            
            v = y(time) - yhat(time);
            if v>theta
                stimes1a(time) = 1;
                % if a spike: add to approximation
                deltaSpike = time - prevspike;
                
                lenK = length(yhat(time:end));
                
                isifact  = mISI;

                dyhat = theta*isifact;
                yhat(time) = yhat(time) + 2*dyhat;

                
                fmult = 2*theta/theta0;

                
                spike(time) = spike(time) + 1;
                spiketime = [spiketime time];
                
                % update theta components:
                th1 = th1 + fmult*thMag;
                
                prevspike = time;
            end
            % evolve yhat
            yhat(time+1) = yhat(time)*dEta;
            spike(time+1) = spike(time)*dEta;
            th1 = th1*dth1;
           
            theta = th1 + theta0;
            thetaTrack(time) = theta;
            
            epsif = epsif+(1-a1)*(spike(time)-epsif);
            epsifTrack(time)=epsif;
            
            
        end
        
        
        
        
    end
    
    
firingratesnn(imnob,:)=tfun;
variancesnn(imnob,:)=tfunmf(:,2);
meansnn(imnob,:)=tfunmf(:,1);

end



subplot(2,1,1)
plot(t,yhat(1:end-1))
hold on

subplot(2,1,2)
plot(t,2*thetaTrack)
hold on


%%
x = 0.01:0.01:5;
h         = single(1);
m_f       = single(0.1);
tau_gamma = single(15);
tau_eta   = single(50);
theta0    = single(0.1);

c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;

c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1);

y1 = x;
y1(x<theta0/2)= 0;
y1 = max(0,h./(exp((c1*y1 + c2)./(c3*y1 + c4)) - 1) - c0 + h/2);

% figure
plot(x,y1,'--b')
hold on
% nex tf
h         = single(1);
tau_gamma = single(15);
tau_eta   = single(50);
theta0    = single(0.1)/2;
m_f       = single(0.0025)/theta0;

c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + (2*m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;

c0 = h/(exp((c1*theta0 + c2)/(c3*theta0 + c4)) - 1);

y2 = x;
y2(x<theta0)= 0;
y2 = max(0,h./(exp((c1*y2 + c2)./(c3*y2 + c4)) - 1) - c0 + h/2);

plot(x,y2,'r')








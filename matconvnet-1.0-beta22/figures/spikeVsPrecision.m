clear all
close all

t=1:1000;
% define thresholdkernel
thMag1 = 0.1;
thTau1 = 15;
etaTau = 50;

theta0v = [0.3 0.05];
mnob = [theta0v(1)*theta0v(1) theta0v(2)*theta0v(2)];

%
% % keep track of assigned theta:
thetaTrack = theta0v'*ones(size(t));
% define eta kernel
etaKernel = exp(-(t-1)./etaTau);


thCorr = 1/(thMag1);
thMag1c = thMag1*thCorr;

dth1 = exp(-1/thTau1);

dEta = exp(-1/etaTau);
eta = 1;

% signal:
tstep = 200;
input = zeros(1,length(t));
input(tstep:end-200) = 1;

yhat = zeros(size(input));
spike = zeros(size(input));
stimes1a = zeros(size(input));
y = input;


tauSig2 = .005;
a2 = exp (-0.001/tauSig2);

epsif = 0;
epsifTrack = ones(size(t));

mISI = 1.25;
ISI = 55;
prevspike = 0;

 
yhatMat = zeros(2,length(t));
thetaTrackMat = zeros(2,length(t));
stimes1aMat = zeros(2,length(t));


yhatpost = zeros(1,length(t));
yhatpostMat = zeros(2,length(t));

epsiMat = zeros(2,length(t));

for ifig = 1:length(mnob)
    
    thMag = mnob(ifig);
    theta0 = theta0v(ifig);
    
    yhat = zeros(size(input));
    stimes1a = zeros(size(input));
    epsif = 0;
    epsifTrack = ones(size(t));
    thetaTrack = ones(size(t))*theta0;
    theta = 1*theta0;
    spike = zeros(size(input));
    yhatpost = zeros(1,length(t));


    mISI = 1.25;
    ISI = 55;
    prevspike = 0;
    
    th1=0;
    

    for time=1:length(t)
        
      
        v = y(time) - yhat(time);
        if v>(1*theta/2)
            stimes1a(time) = 1;
            % if a spike: add to approximation
            deltaSpike = time - prevspike;
            
            lenK = length(yhat(time:end));
            % make update to yhat dependent on isi: larger isi, bigger update
            
            isifact = mISI;
            
            %yhat evolves
            dyhat = theta*isifact;
            yhat(time) = yhat(time) + dyhat;
            % theta(time:end) = theta(time:end) + thMag*thKernel(1:lenK);
            fmult = theta/theta0;
            
            spike(time) = spike(time) + 1;

            % update theta components:
            th1 = th1 + fmult*thMag;
            
            prevspike = time;
        end
        % evolve yhat
        yhat(time+1) = yhat(time)*dEta;
        th1 = th1*dth1;
        
        theta = th1 + theta0;
        thetaTrack(time) = theta;
        spike(time+1) = spike(time)*dEta;
        yhatpost(time+1) = yhatpost(time) + (1-a2)*(spike(time)-yhatpost(time));
        
        epsif = epsif+(1-a2)*(yhat(time)-epsif);
        epsifTrack(time)=epsif;
        
        
    end
    
    
    yhatMat(ifig,:) = yhat(1:time);
    thetaTrackMat(ifig,:) = thetaTrack;
    stimes1aMat(ifig,:) = stimes1a;
    epsiMat(ifig,:) = epsifTrack;
    
    yhatpostMat(ifig,:) = yhatpost(1:time)*thMag;
    
   
end

%%
close all
figure
subplot(2,1,1)

dt = t;
plot(dt,y(dt),'r-',...
    'LineWidth',1.5);
hold on
plot(dt,epsiMat(1,:),'k-',...
    'LineWidth',1.5);
plot(dt,epsiMat(2,:),'b-',...
    'LineWidth',1.5);

ax=gca;
set(ax, ...
    'XLim'        , [0 dt(end)],...
    'YLim'        , [0 1.5],...
    'XTick'       , 0:500:dt(end) , ...
    'LineWidth'   , 1.5         );
set(gca,'FontName','Calibri','FontSize',18,'FontWeight','Normal');

subplot(2,1,2)

stimes1aN=stimes1aMat(1,:);
stimes1aN(stimes1aN==0)=NaN;


stem(dt,0.25*stimes1aN,'k',...
    'LineWidth',1.5);

hold on

stimes1aN=stimes1aMat(2,:);
stimes1aN(stimes1aN==0)=NaN;


stem(dt,0.15*stimes1aN,'b',...
    'LineWidth',1.5);

ax=gca;
set(ax, ...
    'XLim'        , [0 dt(end)],...
    'YLim'        , [0 0.5],...
    'XTick'       , 0:500:dt(end) , ...
    'LineWidth'   , 1.5         );


set(gca,'FontName','Calibri','FontSize',18,'FontWeight','Normal');
hXLabel = xlabel('Time (ms)','FontName','Calibri','FontSize',18,'FontWeight','Normal');